"""
Vector Database Builder Package
================================

Manages vector embeddings, semantic search, and document indexing
for the legal FIR system.
"""

from .vector_store import VectorStore
from .embeddings import EmbeddingGenerator
from .indexer import DocumentIndexer

__all__ = ["VectorStore", "EmbeddingGenerator", "DocumentIndexer"]
