"""
Vector Database Integration with Streamlit App
================================================

Helper functions to integrate vector database with the main app.
"""

import logging
from typing import Optional, List, Dict
from .builder import VectorDatabaseBuilder

logger = logging.getLogger(__name__)

# Global vector database instance
_vector_db_instance: Optional[VectorDatabaseBuilder] = None


def get_vector_db() -> VectorDatabaseBuilder:
    """Get or initialize the global vector database instance."""
    global _vector_db_instance
    
    if _vector_db_instance is None:
        try:
            _vector_db_instance = VectorDatabaseBuilder(
                embedding_backend="huggingface",
                vector_backend="faiss",
                index_path="./vector_indexes",
            )
            
            # Try to load existing indexes
            if _vector_db_instance.load_indexes():
                logger.info("Loaded existing vector database indexes")
            else:
                logger.info("Building new vector database from knowledge base")
                _vector_db_instance.build_from_knowledge_base()
        
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            return None
    
    return _vector_db_instance


def search_ipc_sections_with_vectors(
    complaint_text: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    Search for IPC sections using vector similarity.
    
    Args:
        complaint_text: Complaint text to search with
        top_k: Number of results to return
        
    Returns:
        List of matching IPC sections with similarity scores
    """
    db = get_vector_db()
    if not db:
        logger.error("Vector database not available")
        return []
    
    try:
        return db.search_ipc_sections(complaint_text, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching IPC sections: {e}")
        return []


def search_precedents_with_vectors(
    complaint_text: str,
    top_k: int = 3,
) -> List[Dict]:
    """
    Search for legal precedents using vector similarity.
    
    Args:
        complaint_text: Complaint text to search with
        top_k: Number of results to return
        
    Returns:
        List of matching precedents with similarity scores
    """
    db = get_vector_db()
    if not db:
        logger.error("Vector database not available")
        return []
    
    try:
        return db.search_precedents(complaint_text, top_k=top_k)
    except Exception as e:
        logger.error(f"Error searching precedents: {e}")
        return []


def find_similar_complaints(
    complaint_text: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Find similar past complaints.
    
    Args:
        complaint_text: Complaint text to search with
        top_k: Number of results to return
        
    Returns:
        List of similar complaints with similarity scores
    """
    db = get_vector_db()
    if not db:
        logger.error("Vector database not available")
        return []
    
    try:
        return db.search_similar_complaints(complaint_text, top_k=top_k)
    except Exception as e:
        logger.error(f"Error finding similar complaints: {e}")
        return []


def rebuild_vector_database():
    """Rebuild the vector database from scratch."""
    global _vector_db_instance
    
    try:
        if _vector_db_instance:
            _vector_db_instance.clear_indexes()
        
        # Create new instance
        _vector_db_instance = VectorDatabaseBuilder(
            embedding_backend="huggingface",
            vector_backend="faiss",
            index_path="./vector_indexes",
        )
        
        # Build from knowledge base
        _vector_db_instance.build_from_knowledge_base()
        
        # Try to build from generated FIRs if available
        _vector_db_instance.build_from_generated_firs()
        
        logger.info("Vector database rebuilt successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error rebuilding vector database: {e}")
        return False


def get_vector_db_stats() -> Dict:
    """Get statistics about the vector database."""
    db = get_vector_db()
    if not db:
        return {"error": "Vector database not available"}
    
    try:
        return db.get_database_stats()
    except Exception as e:
        logger.error(f"Error getting vector database stats: {e}")
        return {"error": str(e)}
