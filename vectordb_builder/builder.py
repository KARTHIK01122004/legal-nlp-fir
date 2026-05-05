"""
Vector Database Builder - Main Configuration
==============================================

Configuration and initialization for vector database system.
"""

import logging
import os
from typing import Optional
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .indexer import DocumentIndexer

logger = logging.getLogger(__name__)


class VectorDatabaseBuilder:
    """Main interface for vector database system."""
    
    def __init__(
        self,
        embedding_backend: str = "huggingface",
        vector_backend: str = "faiss",
        index_path: str = "./vector_indexes",
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the vector database builder.
        
        Args:
            embedding_backend: Backend for embeddings (huggingface, openai, ollama)
            vector_backend: Backend for vector storage (faiss, chromadb, pinecone)
            index_path: Path to store vector indexes
            embedding_model: Specific embedding model to use
        """
        self.embedding_backend = embedding_backend
        self.vector_backend = vector_backend
        self.index_path = index_path
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            backend=embedding_backend,
            model_name=embedding_model,
        )
        
        embedding_dim = self.embedding_generator.get_embedding_dimension()
        
        self.vector_store = VectorStore(
            backend=vector_backend,
            index_path=index_path,
            embedding_dim=embedding_dim,
        )
        
        self.indexer = DocumentIndexer(
            vector_store=self.vector_store,
            embedding_generator=self.embedding_generator,
        )
        
        logger.info(f"Initialized VectorDatabaseBuilder:")
        logger.info(f"  Embedding: {embedding_backend} (dim: {embedding_dim})")
        logger.info(f"  Storage: {vector_backend}")
        logger.info(f"  Path: {index_path}")
    
    def build_from_knowledge_base(self):
        """Build vector database from knowledge base."""
        try:
            from knowledge_base.ipc_sections import IPC_SECTIONS
            from knowledge_base.precedents import PRECEDENTS
            
            logger.info("Building vector database from knowledge base...")
            
            # Index IPC sections
            ipc_ids = self.indexer.index_ipc_sections(IPC_SECTIONS)
            logger.info(f"Indexed {len(ipc_ids)} IPC sections")
            
            # Index precedents
            precedent_ids = self.indexer.index_precedents(PRECEDENTS)
            logger.info(f"Indexed {len(precedent_ids)} legal precedents")
            
            # Save indexes
            self.vector_store.save_index()
            
            return True
        
        except Exception as e:
            logger.error(f"Error building from knowledge base: {e}")
            return False
    
    def build_from_generated_firs(self, fir_file: str = "generated_firs.json"):
        """Build vector database from generated FIRs."""
        try:
            if not os.path.exists(fir_file):
                logger.warning(f"FIR file not found: {fir_file}")
                return False
            
            with open(fir_file, "r") as f:
                firs = json.load(f)
            
            logger.info(f"Indexing {len(firs)} generated FIRs...")
            
            # Convert FIRs to complaint format
            complaints = [
                {
                    "complainant": fir.get("complainant", ""),
                    "location": fir.get("location", ""),
                    "date": fir.get("date", ""),
                    "description": fir.get("description", ""),
                    "ipc_sections": fir.get("ipc_sections", []),
                }
                for fir in firs
            ]
            
            # Index complaints
            complaint_ids = self.indexer.index_complaints(complaints)
            logger.info(f"Indexed {len(complaint_ids)} FIR complaints")
            
            # Save indexes
            self.vector_store.save_index()
            
            return True
        
        except Exception as e:
            logger.error(f"Error building from FIRs: {e}")
            return False
    
    def search_ipc_sections(self, query: str, top_k: int = 3) -> list:
        """Search for relevant IPC sections."""
        return self.indexer.find_relevant_ipc_sections(query, top_k=top_k)
    
    def search_precedents(self, query: str, top_k: int = 3) -> list:
        """Search for relevant precedents."""
        return self.indexer.find_relevant_precedents(query, top_k=top_k)
    
    def search_similar_complaints(self, query: str, top_k: int = 5) -> list:
        """Search for similar complaints."""
        return self.indexer.find_similar_complaints(query, top_k=top_k)
    
    def get_database_stats(self) -> dict:
        """Get database statistics."""
        return {
            "embedding_backend": self.embedding_backend,
            "vector_backend": self.vector_backend,
            "indexing_stats": self.indexer.get_indexing_stats(),
        }
    
    def load_indexes(self):
        """Load previously saved indexes."""
        return self.vector_store.load_index()
    
    def clear_indexes(self):
        """Clear all indexes."""
        self.vector_store = VectorStore(
            backend=self.vector_backend,
            index_path=self.index_path,
            embedding_dim=self.embedding_generator.get_embedding_dimension(),
        )
        logger.info("Indexes cleared")


import json
