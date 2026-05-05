"""
IPC Vector Database Builder
============================

Specialized module for building and managing IPC sections vector database.
Handles indexing, updating, and maintaining the IPC knowledge base as vectors.
"""

import logging
import json
import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class IPCVectorDBBuilder:
    """Specialized builder for IPC sections vector database."""
    
    def __init__(
        self,
        embedding_backend: str = "huggingface",
        vector_backend: str = "faiss",
        index_path: str = "./vector_indexes",
    ):
        """
        Initialize IPC vector database builder.
        
        Args:
            embedding_backend: Embedding backend to use
            vector_backend: Vector storage backend
            index_path: Path to store indexes
        """
        self.embedding_backend = embedding_backend
        self.vector_backend = vector_backend
        self.index_path = index_path
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(backend=embedding_backend)
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        self.vector_store = VectorStore(
            backend=vector_backend,
            index_path=index_path,
            embedding_dim=embedding_dim,
        )
        
        self.ipc_sections = {}  # Cache of indexed sections
        self.metadata = {}  # Metadata for tracking
        
        logger.info(f"Initialized IPCVectorDBBuilder with {embedding_backend}/{vector_backend}")
    
    def index_ipc_sections(self, ipc_sections: List[Dict]) -> Dict:
        """
        Index IPC sections into the vector database.
        
        Args:
            ipc_sections: List of IPC section dictionaries
            
        Returns:
            Dictionary with indexing results
        """
        logger.info(f"Indexing {len(ipc_sections)} IPC sections...")
        
        embeddings = []
        documents = []
        metadatas = []
        ids = []
        
        for section in ipc_sections:
            # Create comprehensive text representation
            section_text = self._create_section_text(section)
            documents.append(section_text)
            
            # Store in cache
            self.ipc_sections[section["section"]] = section
            
            # Create metadata
            metadata = {
                "type": "ipc_section",
                "section_number": section["section"],
                "title": section["title"],
                "category": section.get("category", "Unknown"),
                "punishment": section.get("punishment", ""),
                "keywords": section.get("keywords", []),
                "indexed_at": datetime.now().isoformat(),
            }
            metadatas.append(metadata)
            
            ids.append(f"ipc_{section['section']}")
        
        # Generate embeddings for all sections
        logger.info("Generating embeddings...")
        embeddings = self.embedding_generator.generate_batch_embeddings(documents)
        
        # Add to vector store
        logger.info("Adding vectors to store...")
        assigned_ids = self.vector_store.add_vectors(
            embeddings=embeddings,
            documents=documents,
            metadata=metadatas,
            ids=ids,
        )
        
        # Save indexes
        self.vector_store.save_index()
        
        # Track metadata
        self.metadata = {
            "indexed_at": datetime.now().isoformat(),
            "total_sections": len(ipc_sections),
            "embedding_backend": self.embedding_backend,
            "vector_backend": self.vector_backend,
        }
        self._save_metadata()
        
        result = {
            "success": True,
            "indexed_count": len(assigned_ids),
            "ids": assigned_ids,
            "timestamp": datetime.now().isoformat(),
        }
        
        logger.info(f"✓ Successfully indexed {len(assigned_ids)} IPC sections")
        return result
    
    def _create_section_text(self, section: Dict) -> str:
        """Create comprehensive text representation of IPC section."""
        text_parts = [
            f"Section {section['section']}: {section['title']}",
            f"Description: {section.get('description', '')}",
            f"Category: {section.get('category', '')}",
            f"Punishment: {section.get('punishment', '')}",
            f"Keywords: {', '.join(section.get('keywords', []))}",
        ]
        
        return "\n".join(text_parts)
    
    def search_section(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Search for IPC sections by query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of matching sections with scores
        """
        try:
            logger.info(f"Searching IPC database: '{query}'")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            if not query_embedding:
                logger.warning("Could not generate query embedding")
                return []
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=similarity_threshold,
            )
            
            # Format results
            formatted_results = []
            for doc_id, similarity, metadata in results:
                section_num = metadata.get("section_number", "")
                result_dict = {
                    "section": section_num,
                    "title": metadata.get("title", ""),
                    "category": metadata.get("category", ""),
                    "punishment": metadata.get("punishment", ""),
                    "similarity": similarity,
                    "keywords": metadata.get("keywords", []),
                }
                formatted_results.append(result_dict)
            
            logger.info(f"Found {len(formatted_results)} matching IPC sections")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching IPC database: {e}")
            return []
    
    def search_by_category(self, category: str, top_k: int = 10) -> List[Dict]:
        """Search IPC sections by category."""
        results = []
        
        try:
            for section_num, section in self.ipc_sections.items():
                if category.lower() in section.get("category", "").lower():
                    results.append({
                        "section": section_num,
                        "title": section.get("title", ""),
                        "category": section.get("category", ""),
                        "punishment": section.get("punishment", ""),
                    })
            
            results = results[:top_k]
            logger.info(f"Found {len(results)} sections in category '{category}'")
            return results
        
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return []
    
    def search_by_keywords(self, keywords: List[str], top_k: int = 10) -> List[Dict]:
        """Search IPC sections by keywords."""
        results = []
        scores = {}
        
        try:
            for section_num, section in self.ipc_sections.items():
                section_keywords = section.get("keywords", [])
                title = section.get("title", "").lower()
                description = section.get("description", "").lower()
                
                score = 0
                for keyword in keywords:
                    keyword_lower = keyword.lower()
                    if keyword_lower in section_keywords:
                        score += 3
                    if keyword_lower in title:
                        score += 2
                    if keyword_lower in description:
                        score += 1
                
                if score > 0:
                    scores[section_num] = score
                    results.append({
                        "section": section_num,
                        "title": section.get("title", ""),
                        "category": section.get("category", ""),
                        "punishment": section.get("punishment", ""),
                        "keyword_score": score,
                    })
            
            # Sort by score
            results.sort(key=lambda x: x["keyword_score"], reverse=True)
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} sections matching keywords")
            return results
        
        except Exception as e:
            logger.error(f"Error searching by keywords: {e}")
            return []
    
    def get_section_details(self, section_number: str) -> Optional[Dict]:
        """Get full details for a specific IPC section."""
        section_num = str(section_number).strip()
        section = self.ipc_sections.get(section_num)
        
        if section:
            logger.info(f"Retrieved details for Section {section_num}")
            return section
        else:
            logger.warning(f"Section {section_num} not found")
            return None
    
    def get_related_sections(self, section_number: str, top_k: int = 5) -> List[Dict]:
        """Get sections related to a specific section."""
        try:
            section = self.get_section_details(section_number)
            if not section:
                return []
            
            # Search using section description
            query = f"{section['title']} {section['description']}"
            results = self.search_section(query, top_k=top_k + 1)
            
            # Filter out the section itself
            results = [r for r in results if r["section"] != section_number][:top_k]
            
            logger.info(f"Found {len(results)} related sections")
            return results
        
        except Exception as e:
            logger.error(f"Error finding related sections: {e}")
            return []
    
    def update_section(self, section_number: str, updates: Dict) -> bool:
        """Update a specific IPC section."""
        try:
            if section_number not in self.ipc_sections:
                logger.warning(f"Section {section_number} not found")
                return False
            
            # Update cached section
            self.ipc_sections[section_number].update(updates)
            
            # Rebuild indexes for this section
            updated_section = self.ipc_sections[section_number]
            section_text = self._create_section_text(updated_section)
            
            embedding = self.embedding_generator.generate_embedding(section_text)
            if not embedding:
                logger.error("Could not generate embedding for updated section")
                return False
            
            # Update in vector store
            metadata = {
                "type": "ipc_section",
                "section_number": section_number,
                "title": updated_section.get("title", ""),
                "category": updated_section.get("category", ""),
                "updated_at": datetime.now().isoformat(),
            }
            
            self.vector_store.add_vectors(
                embeddings=[embedding],
                documents=[section_text],
                metadata=[metadata],
                ids=[f"ipc_{section_number}"],
            )
            
            self.vector_store.save_index()
            logger.info(f"Updated Section {section_number}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating section: {e}")
            return False
    
    def add_new_section(self, section: Dict) -> bool:
        """Add a new IPC section to the database."""
        try:
            section_num = section.get("section")
            if not section_num:
                logger.error("Section number required")
                return False
            
            if section_num in self.ipc_sections:
                logger.warning(f"Section {section_num} already exists")
                return False
            
            # Index the new section
            result = self.index_ipc_sections([section])
            return result.get("success", False)
        
        except Exception as e:
            logger.error(f"Error adding new section: {e}")
            return False
    
    def delete_section(self, section_number: str) -> bool:
        """Delete an IPC section from the database."""
        try:
            section_num = str(section_number).strip()
            if section_num not in self.ipc_sections:
                logger.warning(f"Section {section_num} not found")
                return False
            
            # Remove from cache
            del self.ipc_sections[section_num]
            
            # Remove from vector store
            self.vector_store.delete_vectors([f"ipc_{section_num}"])
            self.vector_store.save_index()
            
            logger.info(f"Deleted Section {section_num}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting section: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about the IPC vector database."""
        stats = {
            "total_sections": len(self.ipc_sections),
            "embedding_backend": self.embedding_backend,
            "vector_backend": self.vector_backend,
            "vector_store_stats": self.vector_store.get_stats(),
            "last_indexed": self.metadata.get("indexed_at"),
        }
        
        # Count by category
        categories = {}
        for section in self.ipc_sections.values():
            cat = section.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        stats["by_category"] = categories
        
        return stats
    
    def export_index(self, export_path: str) -> bool:
        """Export the index and metadata."""
        try:
            self.vector_store.save_index(export_path)
            
            export_data = {
                "metadata": self.metadata,
                "sections": self.ipc_sections,
                "export_time": datetime.now().isoformat(),
            }
            
            json_path = export_path.replace(".bin", ".json")
            with open(json_path, "w") as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported index to {export_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting index: {e}")
            return False
    
    def _save_metadata(self):
        """Save metadata to file."""
        try:
            metadata_path = os.path.join(self.index_path, "ipc_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")
    
    def load_from_knowledge_base(self):
        """Load IPC sections from knowledge base and index them."""
        try:
            from knowledge_base.ipc_sections import IPC_SECTIONS
            
            logger.info(f"Loading {len(IPC_SECTIONS)} IPC sections from knowledge base...")
            result = self.index_ipc_sections(IPC_SECTIONS)
            
            return result
        
        except ImportError:
            logger.error("Could not import IPC_SECTIONS from knowledge base")
            return {"success": False, "error": "Import failed"}
