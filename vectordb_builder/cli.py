"""
Vector Database CLI Tool
========================

Command-line interface for managing and building the vector database.
Usage: python -m vectordb_builder.cli [command] [options]
"""

import argparse
import logging
import sys
from .builder import VectorDatabaseBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_build(args):
    """Build vector database from knowledge base."""
    logger.info("Building vector database...")
    
    builder = VectorDatabaseBuilder(
        embedding_backend=args.embedding_backend,
        vector_backend=args.vector_backend,
        index_path=args.index_path,
        embedding_model=args.model,
    )
    
    if builder.build_from_knowledge_base():
        logger.info("✓ Vector database built successfully")
        stats = builder.get_database_stats()
        logger.info(f"Stats: {stats}")
        return 0
    else:
        logger.error("✗ Failed to build vector database")
        return 1


def cmd_search(args):
    """Search the vector database."""
    builder = VectorDatabaseBuilder(
        embedding_backend=args.embedding_backend,
        vector_backend=args.vector_backend,
        index_path=args.index_path,
    )
    
    if not builder.load_indexes():
        logger.error("Could not load indexes")
        return 1
    
    logger.info(f"Searching for: {args.query}")
    
    if args.type == "ipc":
        results = builder.search_ipc_sections(args.query, top_k=args.top_k)
        logger.info(f"Found {len(results)} IPC sections:")
        for r in results:
            logger.info(f"  Section {r['section']}: {r['title']} (score: {r['similarity']:.4f})")
    
    elif args.type == "precedent":
        results = builder.search_precedents(args.query, top_k=args.top_k)
        logger.info(f"Found {len(results)} precedents:")
        for r in results:
            logger.info(f"  {r['case_name']} ({r['year']}) - score: {r['similarity']:.4f}")
    
    elif args.type == "complaint":
        results = builder.search_similar_complaints(args.query, top_k=args.top_k)
        logger.info(f"Found {len(results)} similar complaints:")
        for r in results:
            logger.info(f"  {r['complainant']} ({r['location']}) - score: {r['similarity']:.4f}")
    
    return 0


def cmd_stats(args):
    """Display vector database statistics."""
    builder = VectorDatabaseBuilder(
        embedding_backend=args.embedding_backend,
        vector_backend=args.vector_backend,
        index_path=args.index_path,
    )
    
    stats = builder.get_database_stats()
    
    logger.info("Vector Database Statistics:")
    logger.info(f"  Embedding Backend: {stats['embedding_backend']}")
    logger.info(f"  Vector Backend: {stats['vector_backend']}")
    
    indexing_stats = stats.get('indexing_stats', {})
    logger.info(f"  Vector Store Stats: {indexing_stats.get('vector_store_stats', {})}")
    logger.info(f"  Indexed Documents: {indexing_stats.get('indexed_documents', 0)}")
    
    by_type = indexing_stats.get('by_type', {})
    for doc_type, count in by_type.items():
        logger.info(f"    - {doc_type}: {count}")
    
    return 0


def cmd_clear(args):
    """Clear all vector database indexes."""
    if not args.force:
        response = input("Are you sure you want to clear all indexes? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled")
            return 0
    
    builder = VectorDatabaseBuilder(
        embedding_backend=args.embedding_backend,
        vector_backend=args.vector_backend,
        index_path=args.index_path,
    )
    
    builder.clear_indexes()
    logger.info("✓ Indexes cleared")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vector Database Builder for Legal NLP System"
    )
    
    # Global options
    parser.add_argument(
        '--embedding-backend',
        choices=['huggingface', 'openai', 'ollama'],
        default='huggingface',
        help='Embedding backend to use',
    )
    parser.add_argument(
        '--vector-backend',
        choices=['faiss', 'chromadb', 'pinecone'],
        default='faiss',
        help='Vector storage backend to use',
    )
    parser.add_argument(
        '--index-path',
        default='./vector_indexes',
        help='Path to store vector indexes',
    )
    parser.add_argument(
        '--model',
        help='Specific embedding model to use',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Build vector database')
    build_parser.set_defaults(func=cmd_build)
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search vector database')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument(
        '--type',
        choices=['ipc', 'precedent', 'complaint'],
        default='ipc',
        help='Type of documents to search',
    )
    search_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to return',
    )
    search_parser.set_defaults(func=cmd_search)
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.set_defaults(func=cmd_stats)
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear all indexes')
    clear_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt',
    )
    clear_parser.set_defaults(func=cmd_clear)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
