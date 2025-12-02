# Code-RAG MCP Server

## Overview

The Code-RAG MCP (Model Context Protocol) server exposes Code-RAG's semantic code search as tools that Claude can call during a conversation. This lets Claude:
Internally it reuses the same processing, embeddings, and database stack as the CLI, but is wrapped in a dedicated API layer and MCP server.

The **current design** exposes **two tools**:

- **`search_codebase`** – semantic code search with transparent auto-indexing.
- **`get_file_content`** – read file contents with optional line range for deep context.

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

## Available Tools

### Tool 1: `search_codebase`

Semantic search over a local codebase using natural language.

#### Best For
- Discovering where functionality is implemented
- Finding code patterns or examples you don't know the location of
- Understanding how features work across the codebase
- Exploratory searches ("how does X work?")

#### Parameters

- `codebase_path` (required): Absolute path to the codebase root, for example `/home/user/myproject`.
- `query` (required): Natural language description of what you are looking for, such as `"authentication logic"`.
- `max_results` (optional): Number of results to return (default: 5, maximum: 20).
- `show_full_content` (optional): If true, show full chunk content without truncation (default: false).

#### Behavior

1. On the first call when the underlying database is empty, the server silently indexes the given `codebase_path`.
2. Subsequent calls reuse that same index.
3. Results include file paths, line ranges, relevance scores, and code snippets.
4. Use `show_full_content=true` for more context, or use `get_file_content` to read full files.

#### Example Call

```json
{
  "codebase_path": "/home/user/app",
  "query": "database connection logic",
  "max_results": 5,
  "show_full_content": true
}
```

### Tool 2: `get_file_content`

Read the content of a specific file, optionally with a line range.

#### Best For
- Reading full file content after finding it via search
- Getting more context around a specific line range
- Viewing implementation details of a known file

#### Parameters

- `file_path` (required): Absolute path to the file to read.
- `start_line` (optional): Starting line number (1-indexed).
- `end_line` (optional): Ending line number (1-indexed, inclusive).

#### Behavior

1. Returns file content with line numbers prefixed to each line.
2. If no line range is specified, returns the entire file.
3. Invalid line numbers are clamped to valid ranges.

#### Example Call

```json
{
  "file_path": "/home/user/app/src/auth.py",
  "start_line": 10,
  "end_line": 50
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

---

## Typical Workflows

### Search and explore (new codebase)

1. User: "Find the database connection logic in `/home/user/myapp`."
2. Claude calls:
   - `search_codebase(codebase_path="/home/user/myapp", query="database connection logic", max_results=5)`
3. On first use, the MCP server silently indexes `/home/user/myapp`.
4. Claude presents the top matches with file paths, line ranges, and short snippets.

### Get more context after search

1. Claude identifies a relevant file from search results: `src/db/connection.py`
2. Claude calls:
   - `get_file_content(file_path="/home/user/myapp/src/db/connection.py")`
3. Claude can now see the full file context.

### Read specific line range

1. Search results show interesting code at lines 45-78
2. Claude calls:
   - `get_file_content(file_path="/home/user/myapp/src/db/connection.py", start_line=30, end_line=90)`
3. Claude gets surrounding context around the matched code.

---

## Design Notes

### API and MCP layering

- `CodeRAGAPI` (`src/api.py`) provides a clean, reusable façade over the core Code-RAG pipeline.
- `src/mcp_server.py` hosts the MCP server and exposes that API as tools for Claude over stdio.
- `get_file_content` is a simple file reader that doesn't require the full API.

### Two-tool design rationale

The two-tool interface addresses key feedback about the original single-tool design:

1. **Reduced follow-up reads**: Users can use `get_file_content` directly after search.
2. **Better context depth**: The `show_full_content` parameter and line ranges provide more context.
3. **Clear tool separation**: Semantic search for discovery, file reading for deep context.

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
