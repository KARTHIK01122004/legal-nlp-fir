"""
Vector Store
============

Manages vector storage and similarity search.
Supports multiple backends (FAISS, Pinecone, Milvus, ChromaDB).
"""

import logging
import os
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval."""
    
    def __init__(
        self,
        backend: str = "faiss",
        index_path: str = "./vector_indexes",
        embedding_dim: int = 384,
    ):
        """
        Initialize vector store.
        
        Args:
            backend: Storage backend ('faiss', 'pinecone', 'chromadb', 'milvus')
            index_path: Path to store indexes
            embedding_dim: Dimension of embeddings
        """
        self.backend = backend
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata_store = {}  # Store metadata for vectors
        self.id_counter = 0
        
        os.makedirs(index_path, exist_ok=True)
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the vector store backend."""
        try:
            if self.backend == "faiss":
                import faiss
                self.faiss = faiss
                self.index = faiss.IndexFlatL2(self.embedding_dim)
                logger.info("Initialized FAISS vector store")
            
            elif self.backend == "chromadb":
                import chromadb
                self.client = chromadb.Client()
                self.collection = self.client.get_or_create_collection(
                    name="legal_documents",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Initialized ChromaDB vector store")
            
            elif self.backend == "pinecone":
                from pinecone import Pinecone
                self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
                self.index = self.pc.Index("legal-documents")
                logger.info("Initialized Pinecone vector store")
            
            elif self.backend == "milvus":
                from pymilvus import Collection, connections
                connections.connect(host="localhost", port=19530)
                logger.info("Initialized Milvus vector store")
            
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        
        except ImportError as e:
            logger.warning(f"Could not initialize {self.backend}: {e}")
            self.backend = "mock"
    
    def add_vectors(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadata: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add vectors to the store.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of original documents
            metadata: List of metadata dicts (one per embedding)
            ids: Custom IDs for vectors (optional)
            
        Returns:
            List of assigned IDs
        """
        if not embeddings:
            return []
        
        # Generate IDs if not provided
        assigned_ids = []
        if ids is None:
            assigned_ids = [f"doc_{self.id_counter + i}" for i in range(len(embeddings))]
            self.id_counter += len(embeddings)
        else:
            assigned_ids = ids
        
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            if self.backend == "faiss":
                self.index.add(embeddings_array)
                for i, doc_id in enumerate(assigned_ids):
                    self.metadata_store[doc_id] = {
                        "document": documents[i] if i < len(documents) else "",
                        "metadata": metadata[i] if metadata and i < len(metadata) else {},
                        "timestamp": datetime.now().isoformat(),
                    }
                logger.info(f"Added {len(embeddings)} vectors to FAISS index")
            
            elif self.backend == "chromadb":
                self.collection.add(
                    ids=assigned_ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadata or [{}] * len(embeddings),
                )
                logger.info(f"Added {len(embeddings)} vectors to ChromaDB")
            
            elif self.backend == "pinecone":
                vectors_to_upload = []
                for doc_id, embedding, doc, meta in zip(
                    assigned_ids, embeddings, documents, metadata or [{}] * len(embeddings)
                ):
                    vectors_to_upload.append((
                        doc_id,
                        embedding,
                        {**meta, "document": doc}
                    ))
                self.index.upsert(vectors=vectors_to_upload)
                logger.info(f"Added {len(embeddings)} vectors to Pinecone")
            
            else:
                logger.warning(f"Backend '{self.backend}' not fully implemented")
        
        except Exception as e:
            logger.error(f"Error adding vectors: {e}")
            return []
        
        return assigned_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (id, similarity_score, document) tuples
        """
        results = []
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            
            if self.backend == "faiss":
                distances, indices = self.index.search(query_array, top_k)
                for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                    if idx >= 0:  # Valid index
                        similarity = 1 / (1 + dist)  # Convert distance to similarity
                        if similarity >= threshold:
                            doc_id = list(self.metadata_store.keys())[idx] if idx < len(self.metadata_store) else f"doc_{idx}"
                            metadata = self.metadata_store.get(doc_id, {})
                            results.append((doc_id, float(similarity), metadata))
            
            elif self.backend == "chromadb":
                query_result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                )
                for i, doc_id in enumerate(query_result["ids"][0]):
                    distance = query_result["distances"][0][i]
                    similarity = 1 / (1 + distance)
                    if similarity >= threshold:
                        results.append((
                            doc_id,
                            float(similarity),
                            query_result["metadatas"][0][i],
                        ))
            
            elif self.backend == "pinecone":
                query_result = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                )
                for match in query_result.matches:
                    if match.score >= threshold:
                        results.append((match.id, match.score, match.metadata))
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
        
        return results
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        try:
            if self.backend == "chromadb":
                self.collection.delete(ids=ids)
            elif self.backend == "pinecone":
                self.index.delete(ids=ids)
            
            for doc_id in ids:
                self.metadata_store.pop(doc_id, None)
            
            logger.info(f"Deleted {len(ids)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def save_index(self, path: Optional[str] = None):
        """Save index to disk."""
        path = path or os.path.join(self.index_path, f"index_{self.backend}.bin")
        try:
            if self.backend == "faiss":
                self.faiss.write_index(self.index, path)
                
                # Save metadata
                metadata_path = path.replace(".bin", "_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(self.metadata_store, f, indent=2)
            
            logger.info(f"Index saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def load_index(self, path: Optional[str] = None):
        """Load index from disk."""
        path = path or os.path.join(self.index_path, f"index_{self.backend}.bin")
        try:
            if self.backend == "faiss" and os.path.exists(path):
                self.index = self.faiss.read_index(path)
                
                # Load metadata
                metadata_path = path.replace(".bin", "_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        self.metadata_store = json.load(f)
            
            logger.info(f"Index loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        stats = {
            "backend": self.backend,
            "embedding_dimension": self.embedding_dim,
        }
        
        if self.backend == "faiss":
            stats["vector_count"] = self.index.ntotal if self.index else 0
            stats["metadata_count"] = len(self.metadata_store)
        
        return stats
