"""
Document Indexer
================

Indexes legal documents, precedents, and IPC sections for semantic search.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Manages document indexing and retrieval."""
    
    def __init__(self, vector_store, embedding_generator):
        """
        Initialize document indexer.
        
        Args:
            vector_store: VectorStore instance
            embedding_generator: EmbeddingGenerator instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.indexed_documents = {}  # Track indexed docs
        self.document_chunks = {}  # Store document chunks
    
    def index_ipc_sections(self, ipc_sections: List[Dict]) -> List[str]:
        """
        Index IPC sections for semantic search.
        
        Args:
            ipc_sections: List of IPC section dictionaries
            
        Returns:
            List of indexed document IDs
        """
        documents = []
        metadatas = []
        ids = []
        
        for section in ipc_sections:
            # Create text representation
            section_text = f"""
Section {section['section']}: {section['title']}

Description: {section['description']}

Punishment: {section['punishment']}

Keywords: {', '.join(section.get('keywords', []))}

Category: {section.get('category', 'Unknown')}
""".strip()
            
            documents.append(section_text)
            
            metadatas.append({
                "type": "ipc_section",
                "section_number": section["section"],
                "title": section["title"],
                "category": section.get("category", "Unknown"),
                "indexed_at": datetime.now().isoformat(),
            })
            
            ids.append(f"ipc_{section['section']}")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_batch_embeddings(documents)
        
        # Add to vector store
        assigned_ids = self.vector_store.add_vectors(
            embeddings=embeddings,
            documents=documents,
            metadata=metadatas,
            ids=ids,
        )
        
        logger.info(f"Indexed {len(assigned_ids)} IPC sections")
        return assigned_ids
    
    def index_precedents(self, precedents: List[Dict]) -> List[str]:
        """Index legal precedents."""
        documents = []
        metadatas = []
        ids = []
        
        for precedent in precedents:
            # Create text representation
            precedent_text = f"""
Case: {precedent['case_name']}
Year: {precedent['year']}
Court: {precedent['court']}

Relevant Sections: {', '.join(precedent.get('relevant_sections', []))}

Summary: {precedent.get('summary', '')}

Verdict: {precedent.get('verdict', '')}

IPC Application: {', '.join(precedent.get('ipc_application', []))}
""".strip()
            
            documents.append(precedent_text)
            
            metadatas.append({
                "type": "legal_precedent",
                "case_name": precedent["case_name"],
                "year": precedent["year"],
                "court": precedent["court"],
                "relevant_sections": precedent.get("relevant_sections", []),
                "indexed_at": datetime.now().isoformat(),
            })
            
            case_id = precedent["case_name"].lower().replace(" ", "_")
            ids.append(f"precedent_{case_id}")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_batch_embeddings(documents)
        
        # Add to vector store
        assigned_ids = self.vector_store.add_vectors(
            embeddings=embeddings,
            documents=documents,
            metadata=metadatas,
            ids=ids,
        )
        
        logger.info(f"Indexed {len(assigned_ids)} legal precedents")
        return assigned_ids
    
    def index_complaints(self, complaints: List[Dict]) -> List[str]:
        """Index past complaints for similarity matching."""
        documents = []
        metadatas = []
        ids = []
        
        for i, complaint in enumerate(complaints):
            # Create text representation
            complaint_text = f"""
Complainant: {complaint.get('complainant', '')}
Location: {complaint.get('location', '')}
Date: {complaint.get('date', '')}

Description: {complaint.get('description', '')}

Identified Sections: {', '.join(complaint.get('ipc_sections', []))}
""".strip()
            
            documents.append(complaint_text)
            
            metadatas.append({
                "type": "complaint",
                "complainant": complaint.get("complainant", ""),
                "location": complaint.get("location", ""),
                "date": complaint.get("date", ""),
                "ipc_sections": complaint.get("ipc_sections", []),
                "indexed_at": datetime.now().isoformat(),
            })
            
            ids.append(f"complaint_{i}")
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_batch_embeddings(documents)
        
        # Add to vector store
        assigned_ids = self.vector_store.add_vectors(
            embeddings=embeddings,
            documents=documents,
            metadata=metadatas,
            ids=ids,
        )
        
        logger.info(f"Indexed {len(assigned_ids)} complaints")
        return assigned_ids
    
    def search_similar_documents(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            doc_type: Filter by document type (ipc_section, precedent, complaint)
            top_k: Number of results to return
            
        Returns:
            List of (id, similarity, metadata) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            if not query_embedding:
                logger.warning("Could not generate query embedding")
                return []
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Get more results for filtering
            )
            
            # Filter by document type if specified
            if doc_type:
                results = [
                    r for r in results
                    if r[2].get("type") == doc_type
                ]
            
            # Return top_k results
            results = results[:top_k]
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def find_relevant_ipc_sections(self, complaint_text: str, top_k: int = 3) -> List[Dict]:
        """Find relevant IPC sections for a complaint."""
        results = self.search_similar_documents(
            query=complaint_text,
            doc_type="ipc_section",
            top_k=top_k,
        )
        
        sections = []
        for doc_id, similarity, metadata in results:
            sections.append({
                "section": metadata.get("section_number"),
                "title": metadata.get("title"),
                "similarity": similarity,
            })
        
        return sections
    
    def find_relevant_precedents(self, complaint_text: str, top_k: int = 3) -> List[Dict]:
        """Find relevant precedents for a complaint."""
        results = self.search_similar_documents(
            query=complaint_text,
            doc_type="legal_precedent",
            top_k=top_k,
        )
        
        precedents = []
        for doc_id, similarity, metadata in results:
            precedents.append({
                "case_name": metadata.get("case_name"),
                "year": metadata.get("year"),
                "court": metadata.get("court"),
                "similarity": similarity,
            })
        
        return precedents
    
    def find_similar_complaints(self, complaint_text: str, top_k: int = 5) -> List[Dict]:
        """Find similar past complaints."""
        results = self.search_similar_documents(
            query=complaint_text,
            doc_type="complaint",
            top_k=top_k,
        )
        
        similar_complaints = []
        for doc_id, similarity, metadata in results:
            similar_complaints.append({
                "complainant": metadata.get("complainant"),
                "location": metadata.get("location"),
                "similarity": similarity,
                "ipc_sections": metadata.get("ipc_sections", []),
            })
        
        return similar_complaints
    
    def chunk_large_document(
        self,
        document: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """
        Split large document into chunks for better indexing.
        
        Args:
            document: Large document text
            chunk_size: Characters per chunk
            overlap: Character overlap between chunks
            
        Returns:
            List of document chunks
        """
        chunks = []
        start = 0
        
        while start < len(document):
            end = min(start + chunk_size, len(document))
            chunk = document[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks
    
    def get_indexing_stats(self) -> Dict:
        """Get statistics about indexed documents."""
        stats = {
            "vector_store_stats": self.vector_store.get_stats(),
            "indexed_documents": len(self.indexed_documents),
            "document_chunks": len(self.document_chunks),
        }
        
        # Count by document type
        type_counts = {}
        for doc_id, metadata in self.indexed_documents.items():
            doc_type = metadata.get("type", "unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        stats["by_type"] = type_counts
        
        return stats
