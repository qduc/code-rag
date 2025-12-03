# Code-RAG: Quick Start for Developers

## What is Code-RAG?

Code-RAG is a CLI tool that makes codebases searchable using semantic search. It converts source code into vector embeddings, stores them in a database, and lets you query them using natural language.

**Example**: Instead of grepping for function names, ask "authentication logic" and find all relevant auth code.

## Architecture Overview

```
┌─────────────────┐
│  CLI / MCP      │  Entry points (command line + AI assistant integration)
└────────┬────────┘
         │
┌────────▼────────┐
│  File Processor │  Discovers files → Chunks code → Yields metadata
└────────┬────────┘
         │
┌────────▼────────┐
│  Embedding      │  Converts text chunks → vectors (pluggable models)
└────────┬────────┘
         │
┌────────▼────────┐
│  Database       │  Stores vectors + metadata (ChromaDB or Qdrant)
└─────────────────┘
```

**Key Design**: Plugin architecture. Database and embedding implementations are swappable via configuration.

## Components

### 1. File Processor
- **What**: Finds source files, reads them, chunks them
- **How**: Respects `.gitignore`, uses syntax-aware chunking when possible (falls back to line-based)
- **Output**: Text chunks with metadata (file path, chunk position)

### 2. Database Layer
- **Interface**: Abstract base class for vector storage
- **Implementations**: ChromaDB (default, embedded) or Qdrant (networked)
- **Operations**: Store embeddings, similarity search, state tracking

### 3. Embedding Layer
- **Interface**: Abstract base class for text → vector conversion
- **Models**:
  - `all-MiniLM-L6-v2` (default, general purpose)
  - `CodeRankEmbed` (code-optimized)
  - OpenAI's `text-embedding-3-small`
- **Pattern**: Stateless, reusable across all operations

### 4. Configuration
Environment variables control defaults:
- `CODE_RAG_DATABASE_TYPE`: "chroma" or "qdrant"
- `CODE_RAG_EMBEDDING_MODEL`: Model name
- `CODE_RAG_CHUNK_SIZE`: Characters per chunk
- See `src/config/config.py` for full list

### 5. Entry Points
- **CLI** (`code-rag`): Interactive query session
- **MCP Server** (`code-rag-mcp`): Exposes search to AI assistants (Claude, etc.)

## Quick Start

```bash
# Install
pip install -e .

# Index current directory and start querying
code-rag

# Index specific repo with different database
code-rag --path /path/to/repo --database qdrant

# Force reindexing
code-rag --reindex
```

## Common Tasks

**Add a new embedding model?**
→ Extend `EmbeddingInterface`, add to initialization in CLI

**Add a new database backend?**
→ Extend `DatabaseInterface`, add to initialization in CLI

**Change chunk size or ignore patterns?**
→ Modify configuration or file processor settings

**Add support for new languages?**
→ Extend `SyntaxChunker.LANGUAGE_PACKAGES` with tree-sitter binding

**Add new MCP tools?**
→ Update `list_tools()` and `call_tool()` in `src/mcp_server.py`

## Where to Learn More

- **Implementation details**: See `IMPLEMENTATION.md`
- **Code**: Start with `src/main.py` (CLI orchestration)
- **Tests**: `pytest` to run test suite
- **Questions?**: Read the code - it's well-structured and follows the plugin pattern

## Key Principles

1. **Plugin architecture**: New implementations extend interfaces, wired in at initialization
2. **Idempotency**: Checks if codebase is already processed before re-embedding
3. **Batch processing**: Chunks processed in batches for efficiency
4. **Metadata tracking**: Every chunk knows its source file and position

That's it. Now go build something.
