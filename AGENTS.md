# Code-RAG: Quick Start for Developers

## What is Code-RAG?

Code-RAG is a CLI tool that makes codebases searchable using semantic search. It converts source code into vector embeddings, stores them in a database, and lets you query them using natural language.

**Example**: Instead of grepping for function names, ask "authentication logic" and find all relevant auth code.

## Architecture Overview

```
          ┌─────────────┐
          │  CLI / MCP  │  Entry points
          └──────┬──────┘
                 │
          ┌──────▼──────┐
          │    API      │  Orchestration layer (CodeRAGAPI)
          └──────┬──────┘
                 │
    ┌────────────┼────────────┬────────────┐
    │            │            │            │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│Process│    │Search │    │Manage │    │ Embed │
└───┬───┘    └───┬───┘    └───┬───┘    └───┬───┘
    │            │            │            │
┌───▼───┐    ┌───▼───┐    ┌───▼───┐    ┌───▼───┐
│Chunker│    │Rerank │    │Index  │    │Storage│
└───────┘    └───────┘    └───────┘    └───────┘
```

**Key Design**: Orchestrated Plugin architecture. The `CodeRAGAPI` centralizes logic, while specialized components handle chunking, indexing, search analysis, and storage.

## Components

### 1. API Layer (`src/code_rag/api.py`)
- **What**: The central hub for all Code-RAG operations.
- **How**: Integrates embedding, database, reranking, and indexing logic. Used by both CLI and MCP.
- **Features**: Session tracking, auto-generated collection names, and unified indexing flow.

### 2. File Processor & Chunker
- **What**: Discovers source files and breaks them into logical chunks.
- **How**: Uses `SyntaxChunker` (tree-sitter based) for code-aware splitting, falling back to line-based.
- **Output**: Text chunks with rich metadata (file path, line numbers, symbol names).

### 3. Metadata Index (`src/code_rag/index/metadata_index.py`)
- **What**: Tracks state of indexed files for incremental updates.
- **How**: Stores `mtime`, `size`, and `sha256` hashes.
- **Benefit**: Only re-indexes modified files, significantly speeding up subsequent runs.

### 4. Hybrid Search & Query Analyzer
- **What**: Improves search relevance by combining vector search with exact identifier matching.
- **How**: `QueryAnalyzer` detects code identifiers (CamelCase, snake_case) in queries and boosts results containing those identifiers.

### 5. Semantic Reranker (`src/code_rag/reranker/`)
- **What**: Refines search results using Cross-Encoder models.
- **How**: Re-scores top-K candidates from vector search for higher precision.
- **Models**: Defaults to `jinaai/jina-reranker-v3`.

### 6. Embedding & Database Layer
- **Embeddings**: Swappable backends (SentenceTransformers, OpenAI, or Shared HTTP).
- **Databases**: ChromaDB (default) or Qdrant.
- **Features**: Automatic dimension mismatch handling and model idle timeouts.

### 7. Shared Embedding & Reranking Server
- **What**: FastAPI-based server that hosts both embedding and reranker models.
- **Why**: Prevents multiple MCP instances from each loading ~500MB+ of models into RAM.
- **Management**: Auto-spawns on first request, auto-terminates after idle timeout, uses heartbeats.
- **Files**: `src/code_rag/embedding_server.py`, `http_embedding.py`, `http_reranker.py`.

## Quick Start

```bash
# Install
pip install -e .

# Run MCP server (default)
code-rag

# Index and start querying via CLI
code-rag-cli

# Index specific repo with different database
code-rag-cli --path /path/to/repo --database qdrant

# Force reindexing
code-rag-cli --reindex
```

## Common Tasks

**Add a new embedding model?**
→ Extend `EmbeddingInterface`, add to `CodeRAGAPI._create_embedding_model`

**Add a new reranker?**
→ Extend `RerankerInterface`, update `CodeRAGAPI.__init__`

**Adjust reindexing behavior?**
→ Modify `CODE_RAG_REINDEX_DEBOUNCE_MINUTES` or `CODE_RAG_VERIFY_CHANGES_WITH_HASH` in config.

**Change identifier boosting?**
→ Update `QueryAnalyzer.get_boost_score` in `src/code_rag/search/query_analyzer.py`

**Add support for new languages?**
→ Extend `SyntaxChunker.LANGUAGE_PACKAGES` with tree-sitter bindings.

**Add new MCP tools?**
→ Update `list_tools()` and `call_tool()` in `src/code_rag/mcp_server.py`

## Where to Learn More

- **Implementation details**: See `IMPLEMENTATION.md`
- **Code**: Start with `src/code_rag/main.py` (CLI orchestration)
- **Tests**: `pytest` to run test suite
- **Questions?**: Read the code - it's well-structured and follows the plugin pattern

## Key Principles

1. **Plugin architecture**: New implementations extend interfaces, wired in at initialization
2. **Idempotency**: Checks if codebase is already processed before re-embedding
3. **Batch processing**: Chunks processed in batches for efficiency
4. **Metadata tracking**: Every chunk knows its source file and position

That's it. Now go build something.
