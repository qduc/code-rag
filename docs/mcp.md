# Code-RAG MCP Server

## Overview

The Code-RAG MCP (Model Context Protocol) server exposes Code-RAG's semantic code search as tools that Claude can call during a conversation. This lets Claude:
Internally it reuses the same processing, embeddings, and database stack as the CLI, but is wrapped in a dedicated API layer and MCP server.

The **final design** intentionally exposes **one tool only**:

- **`search_codebase`** – semantic code search with transparent auto-indexing.

---

## Installation

From the project root:

```bash
cd /ub/home/qduc/src/code-rag
pip install -e .
```

This installs:

- `code-rag` (CLI)
- `code-rag-mcp` (MCP server entry point)
- All required dependencies, including `mcp>=0.9.0`

Verify that the MCP server is on your PATH:

```bash
which code-rag-mcp
```

---

## Registering with Claude Code

The MCP server uses stdio transport.

### Using the Claude CLI

```bash
claude mcp add code-rag \
  --type stdio \
  -- code-rag-mcp
```

If necessary, you can register it with an explicit path:

```bash
claude mcp add code-rag \
  --type stdio \
  -- /ub/home/qduc/src/code-rag/.venv/bin/code-rag-mcp
```

### Manual configuration (JSON)

A minimal entry in Claude Code's MCP configuration looks like:

```json
{
  "mcpServers": {
    "code-rag": {
      "command": "code-rag-mcp",
      "args": [],
      "env": {
        "CODE_RAG_DATABASE_TYPE": "chroma",
        "CODE_RAG_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
        "CODE_RAG_RERANKER_ENABLED": "true"
      }
    }
  }
}
```

Restart Claude Code after adding the server so it picks up the new configuration.

---

## The One Tool: `search_codebase`

The MCP surface is deliberately minimal: **one tool that does one thing well**.

### Purpose

- Semantic search over a local codebase using natural language.
- Handles **indexing automatically on first use**; no separate indexing commands.

### Parameters

- `codebase_path` (required): Absolute path to the codebase root, for example `/home/user/myproject`.
- `query` (required): Natural language description of what you are looking for, such as `"authentication logic"`.
- `max_results` (optional): Number of results to return (default: 5, maximum: 20).

### Behavior

1. On the first call when the underlying database is empty, the server silently indexes the given `codebase_path`.
2. Subsequent calls reuse that same index. The current implementation maintains a *single* shared index; changing `codebase_path` while reusing the same database path/server process does **not** create a new index.
3. Results include file paths, line ranges, relevance scores, and a short snippet for each hit.
4. Claude can then use its own file-reading tools (e.g., `Read`) to fetch full files or larger ranges.

### Example Call (conceptual)

```json
{
  "codebase_path": "/home/user/app",
  "query": "database connection logic",
  "max_results": 5
}
```

---

## Configuration

The MCP server shares configuration with the CLI via environment variables:

| Variable | Description | Example |
|---------|-------------|---------|
| `CODE_RAG_DATABASE_TYPE` | Vector database backend (`chroma` or `qdrant`) | `chroma` |
| `CODE_RAG_DATABASE_PATH` | Database storage directory on disk | `$HOME/.cache/code-rag` |
| `CODE_RAG_EMBEDDING_MODEL` | Embedding model identifier | `sentence-transformers/all-MiniLM-L6-v2` or `text-embedding-3-small` |
| `CODE_RAG_CHUNK_SIZE` | Characters per chunk | `1024` |
| `CODE_RAG_BATCH_SIZE` | Batch size when processing | `32` |
| `CODE_RAG_RERANKER_ENABLED` | Enable semantic reranking | `true` / `false` |
| `CODE_RAG_RERANKER_MODEL` | Cross-encoder reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CODE_RAG_RERANKER_MULTIPLIER` | Retrieval multiplier before reranking | `2` |

Note: The Code-RAG CLI uses these variables for embedding, chunking, batching,
and reranking (for example, `CODE_RAG_EMBEDDING_MODEL`,
`CODE_RAG_CHUNK_SIZE`, `CODE_RAG_BATCH_SIZE`, `CODE_RAG_RERANKER_ENABLED`,
`CODE_RAG_RERANKER_MODEL`, and `CODE_RAG_RERANKER_MULTIPLIER`). The MCP
server currently reads `CODE_RAG_DATABASE_PATH`, `CODE_RAG_CHUNK_SIZE`,
`CODE_RAG_BATCH_SIZE`, and `CODE_RAG_RERANKER_MODEL`. The
`CODE_RAG_DATABASE_TYPE` variable is defined for future use; at the moment the
CLI selects the database via its `--database` flag and the MCP server always
uses the Chroma backend.

Examples when registering via CLI:

```bash
claude mcp add code-rag \
  --type stdio \
  --env CODE_RAG_DATABASE_TYPE=qdrant \
  --env CODE_RAG_EMBEDDING_MODEL=text-embedding-3-small \
  -- code-rag-mcp
```

To use OpenAI embeddings (CLI only):

```bash
export OPENAI_API_KEY="sk-..."

claude mcp add code-rag \
  --type stdio \
  --env CODE_RAG_EMBEDDING_MODEL=text-embedding-3-small \
  --env OPENAI_API_KEY=$OPENAI_API_KEY \
  -- code-rag-mcp

Note: The CLI respects `CODE_RAG_EMBEDDING_MODEL` and will switch to
OpenAI embeddings when the name starts with `text-embedding-`. The MCP
server currently always uses the default local sentence-transformers
model, regardless of this variable.
```

---

## Typical Workflows

### Search a codebase (indexing happens automatically)

1. User: "Find the database connection logic in `/home/user/myapp`."
2. Claude calls:
   - `search_codebase(codebase_path="/home/user/myapp", query="database connection logic", max_results=5)`
3. On first use, the MCP server silently indexes `/home/user/myapp`.
4. Claude presents the top matches with file paths, line ranges, and short snippets.

### Locate where something is implemented

1. User: "Where is the user authentication implemented?"
2. Claude calls:
   - `search_codebase(codebase_path="/home/user/myapp", query="user authentication implementation")`
3. Claude returns locations such as:
   - `src/auth/authenticator.py` (main authentication logic)
   - `src/middleware/auth.py` (authentication middleware)
4. If the user wants more context, Claude uses its own file-reading capability to show full files.

---

## Design Notes and Future Direction

### API and MCP layering

- `CodeRAGAPI` (`src/api.py`) provides a clean, reusable façade over the core Code-RAG pipeline (initialize collection, index, search, get chunks, stats, close).
- `src/mcp_server.py` hosts the MCP server and exposes that API as tools for Claude over stdio.
- The same API can later back other interfaces (e.g., HTTP, gRPC) without changing core logic.

### Single-tool, auto-indexing design (final)

The **single-tool interface** is the final intended surface for the MCP server:

- Only `search_codebase` is exposed.
- Indexing and low-level status become implementation details hidden behind that tool.
- Typical workflow is: *search → (optionally) read full files using Claude's existing tools*.

This keeps the MCP server focused on its unique value (semantic search) while relying on Claude's built-in capabilities for file reading and navigation.

---

## Troubleshooting (Quick Reference)

- **MCP server not found**
  - `which code-rag-mcp` to confirm installation
  - Re-register with full path if needed
  - Restart Claude Code and check MCP logs

- **Import/dependency errors**
  - Ensure the correct virtualenv is active
  - Reinstall: `pip install -e . --force-reinstall`

- **Database issues**
  - Confirm `CODE_RAG_DATABASE_PATH` is writable (or Qdrant is reachable)
  - Try the default `chroma` backend if Qdrant is failing

- **Performance concerns**
  - Tune `CODE_RAG_BATCH_SIZE` and `CODE_RAG_CHUNK_SIZE`
  - Disable reranking for lighter queries: `CODE_RAG_RERANKER_ENABLED=false`
  - Use a lighter embedding model if needed
