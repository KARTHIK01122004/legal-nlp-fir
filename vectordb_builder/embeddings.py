"""
Embedding Generator
===================

Generates and manages text embeddings using multiple backend options.
"""

import logging
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using various backends."""
    
    def __init__(self, backend: str = "huggingface", model_name: Optional[str] = None):
        """
        Initialize embedding generator.
        
        Args:
            backend: Embedding backend ('huggingface', 'openai', 'ollama')
            model_name: Specific model to use
        """
        self.backend = backend
        self.model_name = model_name or self._get_default_model()
        self.model = None
        self._initialize_backend()
    
    def _get_default_model(self) -> str:
        """Get default model for backend."""
        defaults = {
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            "openai": "text-embedding-3-small",
            "ollama": "nomic-embed-text",
        }
        return defaults.get(self.backend, defaults["huggingface"])
    
    def _initialize_backend(self):
        """Initialize the embedding backend."""
        try:
            if self.backend == "huggingface":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Initialized HuggingFace backend with model: {self.model_name}")
            
            elif self.backend == "openai":
                from openai import OpenAI
                self.client = OpenAI()
                logger.info(f"Initialized OpenAI backend with model: {self.model_name}")
            
            elif self.backend == "ollama":
                import ollama
                self.client = ollama
                logger.info(f"Initialized Ollama backend with model: {self.model_name}")
            
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
        
        except ImportError as e:
            logger.error(f"Failed to import backend '{self.backend}': {e}")
            self.backend = "mock"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return []
        
        try:
            if self.backend == "huggingface":
                embedding = self.model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            
            elif self.backend == "openai":
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model_name
                )
                return response.data[0].embedding
            
            elif self.backend == "ollama":
                response = self.client.embeddings(
                    model=self.model_name,
                    prompt=text
                )
                return response["embedding"]
            
            elif self.backend == "mock":
                # Mock embedding for testing
                return [0.0] * 384
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        try:
            if self.backend == "huggingface":
                batch_embeddings = self.model.encode(texts, convert_to_tensor=False)
                embeddings = batch_embeddings.tolist()
            
            elif self.backend == "openai":
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                embeddings = [item.embedding for item in response.data]
            
            elif self.backend == "ollama":
                embeddings = [
                    self.client.embeddings(model=self.model_name, prompt=text)["embedding"]
                    for text in texts
                ]
            
            else:
                embeddings = [[0.0] * 384 for _ in texts]
        
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            embeddings = [[0.0] * 384 for _ in texts]
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings."""
        try:
            if self.backend == "huggingface":
                return self.model.get_sentence_embedding_dimension()
            elif self.backend in ["openai", "ollama"]:
                test_embedding = self.generate_embedding("test")
                return len(test_embedding) if test_embedding else 1536
            else:
                return 384
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return 384
    
    def switch_backend(self, backend: str, model_name: Optional[str] = None):
        """Switch to a different embedding backend."""
        self.backend = backend
        self.model_name = model_name or self._get_default_model()
        self._initialize_backend()
        logger.info(f"Switched to {backend} backend")
