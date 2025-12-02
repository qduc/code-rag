# Code-RAG: Semantic Code Search Tool

Code-RAG is a powerful CLI tool that converts source codebases into searchable vector embeddings using RAG (Retrieval-Augmented Generation). It enables semantic code search through natural language queries, making it easy to find relevant code snippets, understand codebase structure, and analyze patterns across your projects.

## Features

- 🔍 **Semantic Search**: Use natural language to find code (e.g., "authentication logic", "database connection")
- 🚀 **Multiple Databases**: Pluggable database support (ChromaDB, Qdrant)
- 🎯 **Smart Chunking**: Syntax-aware chunking for 40+ programming languages using Tree-sitter
- 🔄 **Flexible Embeddings**: Support for local models (sentence-transformers) and OpenAI embeddings
- ⚡ **Reranking**: Optional semantic reranking for improved search results
- 🤖 **MCP Server**: Integrate with Claude Code as a tool for AI-powered code analysis
- 🎨 **Smart Filtering**: Respects `.gitignore` and common ignore patterns

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd code-rag

# Install in development mode
pip install -e .
```

### Basic Usage

```bash
# Index and search current directory
code-rag

# Index a specific codebase
code-rag --path /path/to/your/project

# Force reindexing
code-rag --reindex

# Use a different embedding model
code-rag --model sentence-transformers/paraphrase-MiniLM-L6-v2

# Use Qdrant instead of ChromaDB
code-rag --database qdrant

# Disable reranking
code-rag --no-rerank

# Change number of results
code-rag --results 10
```

### Example Session

```
$ code-rag --path /home/user/myproject

==============================================================
Code-RAG: Codebase Vector Search Tool
==============================================================
Codebase path: /home/user/myproject
Database type: chroma
Embedding model: sentence-transformers/all-MiniLM-L6-v2
Reranking: enabled
==============================================================

Loading embedding model...
Model loaded. Embedding dimension: 384

Loading reranker model...
Reranker loaded: cross-encoder/ms-marco-MiniLM-L-6-v2

Processing codebase...
Found 247 files to process
Processing (247/247): src/utils/helpers.py

Processing complete! Indexed 1,234 chunks.

==============================================================
Query Session Started
Enter your query to search the codebase.
Press Ctrl+C to exit.
==============================================================

Query: authentication logic

Found 5 results:

------------------------------------------------------------
Result 1 | Similarity: 0.8542
File: src/auth/authenticator.py
Lines: 45-78 | Chunk: 2/5
------------------------------------------------------------
class Authenticator:
    """Handle user authentication and session management."""

    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user credentials."""
        ...
```

## MCP Server Integration

Code-RAG can be used as an MCP (Model Context Protocol) server, allowing Claude to automatically search your codebase during conversations.

### Quick MCP Setup

```bash
# Register with Claude Code
claude mcp add code-rag --type stdio -- code-rag-mcp
```

For detailed MCP setup instructions, see [MCP_SETUP.md](MCP_SETUP.md).

### Using Code-RAG with Claude

Once configured, you can interact with Claude:

```
You: "Index my project at /home/user/myapp and find the database connection logic"

Claude: [Automatically indexes the codebase and searches for database logic]
        "I found the database connection logic in src/db/connection.py..."
```

## Architecture

Code-RAG uses a plugin-based architecture with abstract interfaces for extensibility:

### Components

1. **File Processor**: Discovers, reads, and chunks source files
   - Syntax-aware chunking using Tree-sitter for supported languages
   - Automatic fallback to line-aware chunking
   - Respects `.gitignore` and standard ignore patterns

2. **Database Layer**: Pluggable vector database interface
   - ChromaDB (default): Lightweight embedded database
   - Qdrant: Network-based option for distributed deployments
   - Easy to add new databases

3. **Embedding Layer**: Pluggable embedding model interface
   - Local: sentence-transformers (no API calls)
   - OpenAI: text-embedding-3-small, text-embedding-ada-002
   - Easy to add new providers

4. **Reranker**: Optional semantic reranking
   - Uses cross-encoder models for improved relevance
   - Configurable retrieval multiplier

5. **MCP Server**: Model Context Protocol integration
   - Exposes Code-RAG as tools for Claude
   - Supports indexing, searching, and status checks

## Configuration

Code-RAG can be configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `CODE_RAG_DATABASE_TYPE` | Database type ("chroma" or "qdrant") | `chroma` |
| `CODE_RAG_DATABASE_PATH` | Database storage path | Platform cache directory |
| `CODE_RAG_EMBEDDING_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `CODE_RAG_CHUNK_SIZE` | Characters per chunk | `1024` |
| `CODE_RAG_BATCH_SIZE` | Batch size for processing | `32` |
| `CODE_RAG_RERANKER_ENABLED` | Enable semantic reranking | `true` |
| `CODE_RAG_RERANKER_MODEL` | Reranker model name | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CODE_RAG_RERANKER_MULTIPLIER` | Retrieval multiplier for reranking | `2` |

### Example Configuration

```bash
# Use OpenAI embeddings
export OPENAI_API_KEY="sk-..."
export CODE_RAG_EMBEDDING_MODEL="text-embedding-3-small"

# Use Qdrant database
export CODE_RAG_DATABASE_TYPE="qdrant"
export CODE_RAG_DATABASE_PATH="http://localhost:6333"

# Adjust chunking
export CODE_RAG_CHUNK_SIZE="2048"
export CODE_RAG_BATCH_SIZE="64"

# Run Code-RAG
code-rag --path /path/to/project
```

## Supported Languages

Code-RAG supports syntax-aware chunking for:

- Python
- JavaScript / TypeScript
- Go
- Rust
- Java
- C / C++

For other languages, it automatically falls back to line-aware chunking.

## API Usage

You can also use Code-RAG programmatically:

```python
from src.api import CodeRAGAPI

# Initialize API
api = CodeRAGAPI(
    database_type="chroma",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    reranker_enabled=True
)

# Initialize collection
api.initialize_collection("myproject")

# Index a codebase
def progress(current, total, file_path):
    print(f"Processing {current}/{total}: {file_path}")

chunks = api.index_codebase(
    "/path/to/project",
    progress_callback=progress
)
print(f"Indexed {chunks} chunks")

# Search
results = api.search("authentication logic", n_results=5)
for result in results:
    print(f"{result['file_path']} - Similarity: {result['similarity']:.4f}")
    print(result['content'][:200])
    print()

# Get specific chunk
chunk = api.get_chunk("src/auth/authenticator.py", chunk_index=1)
if chunk:
    print(chunk['content'])

# Check status
print(f"Is processed: {api.is_processed()}")
print(f"Total chunks: {api.count()}")

# Cleanup
api.close()
```

## Development

### Project Structure

```
code-rag/
├── src/
│   ├── main.py              # CLI entry point
│   ├── api.py               # Public API layer
│   ├── mcp_server.py        # MCP server implementation
│   ├── config/
│   │   └── config.py        # Configuration management
│   ├── database/
│   │   ├── database_interface.py   # Abstract database interface
│   │   ├── chroma_database.py      # ChromaDB implementation
│   │   └── qdrant_database.py      # Qdrant implementation
│   ├── embeddings/
│   │   ├── embedding_interface.py          # Abstract embedding interface
│   │   ├── sentence_transformer_embedding.py
│   │   └── openai_embedding.py
│   ├── processor/
│   │   └── file_processor.py        # File discovery and chunking
│   └── reranker/
│       ├── reranker_interface.py    # Abstract reranker interface
│       └── cross_encoder_reranker.py
├── setup.py
├── requirements.txt
├── README.md
├── MCP_SETUP.md
└── AGENTS.md             # Architecture documentation
```

### Testing

```bash
# Run tests
pytest

# Code quality
black src/
flake8 src/
```

### Adding a New Database

1. Create a new file in `src/database/` (e.g., `my_database.py`)
2. Implement the `DatabaseInterface` abstract class
3. Add initialization logic in `src/main.py` and `src/api.py`

### Adding a New Embedding Provider

1. Create a new file in `src/embeddings/` (e.g., `my_embedding.py`)
2. Implement the `EmbeddingInterface` abstract class
3. Add initialization logic in `src/main.py` and `src/api.py`

## Performance Tips

- **Use smaller chunk sizes** for more granular search results
- **Increase batch size** for faster processing (requires more memory)
- **Disable reranking** if search speed is critical
- **Use local models** to avoid API costs and latency
- **Use Qdrant** for distributed deployments or very large codebases

## Troubleshooting

### Import Errors

```bash
# Reinstall with all dependencies
pip install -e . --force-reinstall
```

### Database Issues

```bash
# Clear the database and reindex
code-rag --reindex
```

### Memory Issues

```bash
# Reduce batch size
export CODE_RAG_BATCH_SIZE="16"
code-rag
```

### OpenAI Rate Limits

```bash
# Use local models instead
export CODE_RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
code-rag
```

## Documentation

- [MCP_SETUP.md](MCP_SETUP.md) - MCP server setup and usage
- [AGENTS.md](AGENTS.md) - Architecture and design patterns
- [init.md](init.md) - Original project specification

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Support

For issues or questions:
- Open an issue on GitHub
- Check the troubleshooting section above
- Review the documentation files

## Acknowledgments

- Built with [ChromaDB](https://www.trychroma.com/) and [Qdrant](https://qdrant.tech/)
- Uses [sentence-transformers](https://www.sbert.net/) for embeddings
- Powered by [Tree-sitter](https://tree-sitter.github.io/) for syntax parsing
- MCP integration via [Model Context Protocol](https://modelcontextprotocol.io/)
