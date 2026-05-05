"""
Vector Database README
=====================

Comprehensive documentation for the vector database system.
"""

# Vector Database Builder

The `vectordb_builder` module provides semantic search capabilities for the legal FIR system using vector embeddings and similarity search.

## Features

- **Multi-Backend Support**: FAISS, ChromaDB, Pinecone for vector storage
- **Multiple Embedding Options**: HuggingFace, OpenAI, Ollama
- **Semantic Search**: Find IPC sections, precedents, and similar complaints by meaning
- **Intelligent Indexing**: Automatically index legal documents for fast retrieval
- **CLI Tool**: Command-line interface for database management
- **Streamlit Integration**: Easy integration with the main web application

## Quick Start

### 1. Initialize Vector Database

```python
from vectordb_builder import VectorDatabaseBuilder

# Create builder instance
db = VectorDatabaseBuilder(
    embedding_backend="huggingface",
    vector_backend="faiss",
    index_path="./vector_indexes"
)

# Build from knowledge base
db.build_from_knowledge_base()
```

### 2. Search IPC Sections

```python
# Find relevant IPC sections for a complaint
complaint = "I was robbed at knife-point yesterday"
results = db.search_ipc_sections(complaint, top_k=3)

for result in results:
    print(f"Section {result['section']}: {result['title']}")
    print(f"Similarity: {result['similarity']:.4f}\n")
```

### 3. Search Precedents

```python
# Find similar legal cases
results = db.search_precedents(complaint, top_k=3)

for result in results:
    print(f"{result['case_name']} ({result['year']})")
    print(f"Similarity: {result['similarity']:.4f}\n")
```

### 4. Find Similar Complaints

```python
# Find past similar complaints
results = db.search_similar_complaints(complaint, top_k=5)

for result in results:
    print(f"Complainant: {result['complainant']}")
    print(f"Location: {result['location']}")
    print(f"Similarity: {result['similarity']:.4f}\n")
```

## Using with Streamlit

```python
from vectordb_builder.streamlit_integration import (
    search_ipc_sections_with_vectors,
    search_precedents_with_vectors,
    find_similar_complaints,
)

# In your Streamlit app
complaint_text = st.text_area("Enter complaint:")

if st.button("Search"):
    # Search IPC sections
    ipc_results = search_ipc_sections_with_vectors(complaint_text, top_k=3)
    st.write("Relevant IPC Sections:", ipc_results)
    
    # Search precedents
    precedent_results = search_precedents_with_vectors(complaint_text, top_k=3)
    st.write("Similar Cases:", precedent_results)
    
    # Find similar complaints
    similar = find_similar_complaints(complaint_text, top_k=5)
    st.write("Similar Complaints:", similar)
```

## CLI Commands

### Build Database

```bash
python -m vectordb_builder.cli build
python -m vectordb_builder.cli build --embedding-backend openai --model text-embedding-3-large
```

### Search Database

```bash
python -m vectordb_builder.cli search "robbery with knife" --type ipc --top-k 5
python -m vectordb_builder.cli search "sexual assault case" --type precedent --top-k 3
python -m vectordb_builder.cli search "similar theft complaint" --type complaint --top-k 5
```

### View Statistics

```bash
python -m vectordb_builder.cli stats
```

### Clear Database

```bash
python -m vectordb_builder.cli clear --force
```

## Configuration

### Embedding Backends

1. **HuggingFace** (Default)
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Dimension: 384
   - Requires: `sentence-transformers`

2. **OpenAI**
   - Model: `text-embedding-3-small`
   - Dimension: 1536
   - Requires: API key

3. **Ollama**
   - Model: `nomic-embed-text`
   - Requires: Local Ollama instance

### Vector Storage Backends

1. **FAISS** (Default)
   - File-based storage
   - Fast similarity search
   - Requires: `faiss-cpu` or `faiss-gpu`

2. **ChromaDB**
   - Lightweight SQL database
   - Persistent storage
   - Requires: `chromadb`

3. **Pinecone**
   - Cloud-based vector database
   - Scalable
   - Requires: API key

## Module Structure

```
vectordb_builder/
├── __init__.py                 # Package initialization
├── embeddings.py              # Embedding generation
├── vector_store.py            # Vector storage backends
├── indexer.py                 # Document indexing
├── builder.py                 # Main builder class
├── streamlit_integration.py   # Streamlit helpers
├── cli.py                     # Command-line interface
└── README.md                  # This file
```

## Performance Tips

1. **Batch Processing**: Use `generate_batch_embeddings()` for multiple documents
2. **Caching**: Results are cached automatically
3. **Index Persistence**: Save and load indexes to avoid rebuilding
4. **Dimension Selection**: Lower dimensions = faster search but less accuracy

## Troubleshooting

### Import Errors

If you get import errors for sentence-transformers, install it:
```bash
pip install sentence-transformers
```

For FAISS:
```bash
pip install faiss-cpu  # or faiss-gpu for GPU support
```

### Slow Search

- Reduce `top_k` parameter
- Use faster embedding model
- Ensure indexes are saved and loaded correctly

### Out of Memory

- Process documents in smaller batches
- Use cloud backends (Pinecone, ChromaDB Cloud)
- Reduce embedding dimension

## Future Enhancements

- [ ] Hybrid search (keyword + semantic)
- [ ] Real-time index updates
- [ ] Multi-language support
- [ ] Query expansion
- [ ] Relevance feedback
- [ ] Batch indexing optimization

## License

See main project LICENSE file.
