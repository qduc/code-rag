# Code-RAG

[![PyPI version](https://img.shields.io/pypi/v/code-rag-mcp.svg)](https://pypi.org/project/code-rag-mcp/)
[![Build Status](https://github.com/qduc/code-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/qduc/code-rag/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/code-rag-mcp.svg)](https://pypi.org/project/code-rag-mcp/)

**Semantic code search for your entire codebase.** Ask questions in plain English, get relevant code snippets with source locations.

Instead of grepping for function names, ask "authentication logic" and find all related auth code across your project.

## Table of Contents

- [Quick Start](#quick-start)
- [Example (CLI)](#example-cli)
- [Why Use Code-RAG?](#why-use-code-rag)
- [Use with Claude Code (MCP Integration)](#use-with-claude-code-mcp-integration)
- [Configuration](#configuration)
- [How It Works](#how-it-works)
- [API Usage](#api-usage)
- [Supported Languages](#supported-languages)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)

## Quick Start

```bash
# Create virtual environment, optional but recommended
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Use the interactive CLI search
code-rag-cli

# To use as MCP server see below section
```

## Why Use Code-RAG?

- **Understand unfamiliar codebases** - Ask questions instead of reading everything
- **Find examples** - "error handling with retries" finds all relevant patterns
- **Refactoring aid** - Locate all code related to a feature you're changing
- **Documentation** - Extract context for writing docs or onboarding

## Use with Claude Code (MCP Integration)

Code-RAG works as an MCP server, letting Claude automatically search your codebase during conversations.

> **Note on `uv`:** Many examples below use [uv](https://github.com/astral-sh/uv) (specifically `uvx`) for fast, zero-config execution. If you don't have `uv` installed, you can use standard `pip` or `npx` (if using a wrapper).

### Quick Setup

**Option 1: Using uvx (Recommended)**
```bash
# Install uv first: https://github.com/astral-sh/uv

# Claude Code
claude mcp add code-rag --transport stdio uvx code-rag-mcp
```

**Option 2: Using pip (Standard)**
```bash
# Install in your environment
pip install code-rag-mcp

# Register with Claude Code using the absolute path to the binary
claude mcp add code-rag --transport stdio $(which code-rag-mcp)
```

**Option 3: Local development installation**
```bash
# Clone and install
git clone https://github.com/qduc/code-rag.git
cd code-rag
pip install -e .

# Register with Claude Code
claude mcp add code-rag --transport stdio $(pwd)/.venv/bin/code-rag-mcp
```

### Configuration

The MCP server reads configuration from environment variables or config files. Configure via your MCP client's settings:

**Claude Desktop (`~/Library/Application Support/Claude/claude_desktop_config.json`)**:
```json
{
  "mcpServers": {
    "code-rag": {
      "command": "uvx",
      "args": ["code-rag-mcp"],
      "env": {
        "CODE_RAG_EMBEDDING_MODEL": "nomic-ai/CodeRankEmbed",
        "CODE_RAG_DATABASE_TYPE": "chroma",
        "CODE_RAG_RERANKER_ENABLED": "true"
      }
    }
  }
}
```

**Claude Code (via claude mcp add)**:
```bash
# Basic setup
claude mcp add code-rag --transport stdio uvx code-rag-mcp

# Then configure via environment variables or config files
```

**Common Configuration Options**:
- `CODE_RAG_EMBEDDING_MODEL` - Embedding model (default: `nomic-ai/CodeRankEmbed`)
  - `nomic-ai/CodeRankEmbed` - Code-optimized, runs locally, requires GPU for best performance
  - `text-embedding-3-small` - OpenAI embeddings, no GPU required (requires `OPENAI_API_KEY`)
- `CODE_RAG_DATABASE_TYPE` - Database backend: `chroma` or `qdrant` (default: `chroma`)
- `CODE_RAG_CHUNK_SIZE` - Chunk size in characters (default: `1024`)
- `CODE_RAG_RERANKER_ENABLED` - Enable result reranking, may yield better results but slower (default: `false`)
- `CODE_RAG_SHARED_SERVER` - Share embedding server across instances, reduce memory footprint (default: `true`)

**Example with OpenAI embeddings**:
```json
{
  "mcpServers": {
    "code-rag": {
      "command": "uvx",
      "args": ["code-rag-mcp"],
      "env": {
        "CODE_RAG_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_API_KEY": "sk-...",
        "CODE_RAG_RERANKER_ENABLED": "true"
      }
    }
  }
}
```

### Usage

Once configured, Claude can automatically search your codebase:

```
You: "Find the database connection logic"

Claude: [Automatically searches and finds the code]
        "I found the database connection logic in src/code_rag/db/connection.py..."
```

See [docs/mcp.md](docs/mcp.md) for detailed setup and troubleshooting.

## Basic Usage

```bash
# Different codebase
code-rag-cli --path /path/to/repo

# Force reindex
code-rag-cli --reindex

# More results
code-rag-cli --results 10

# Different embedding model (better for code)
code-rag-cli --model text-embedding-3-small # need to set OPENAI_API_KEY env

# Use Qdrant instead of ChromaDB
code-rag-cli --database qdrant
```

## Configuration

### Priority Order

Configuration is loaded in this order (higher priority overrides lower):

1. **Environment variables** (highest priority)
2. Custom config file via `CODE_RAG_CONFIG_FILE` environment variable
3. Project config: `./code-rag.config`
4. User config: `~/.config/code-rag/config` (auto-created with defaults)

**For MCP servers**: Set environment variables in your MCP client config (see MCP Integration section above).

**For CLI usage**: Use environment variables or config files.

### Environment Variables

```bash
# Use code-optimized embeddings (recommended)
export CODE_RAG_EMBEDDING_MODEL="nomic-ai/CodeRankEmbed"

# Or OpenAI embeddings
export OPENAI_API_KEY="sk-..."
export CODE_RAG_EMBEDDING_MODEL="text-embedding-3-small"

# Use Qdrant
export CODE_RAG_DATABASE_TYPE="qdrant"

# Adjust chunk size
export CODE_RAG_CHUNK_SIZE="2048"

# Enable reranking for better results
export CODE_RAG_RERANKER_ENABLED="true"

# Add custom ignore patterns (comma-separated)
export CODE_RAG_ADDITIONAL_IGNORE_PATTERNS="*.tmp,*.bak,logs/"
```

### Config File Format

Config files use the same format (key=value):

```bash
# ~/.config/code-rag/config or ./code-rag.config
CODE_RAG_EMBEDDING_MODEL=nomic-ai/CodeRankEmbed
CODE_RAG_DATABASE_TYPE=chroma
CODE_RAG_CHUNK_SIZE=1024
CODE_RAG_RERANKER_ENABLED=false
```

Full configuration options in [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md#configuration-system).

## How It Works

1. **Scans** your codebase (respects `.gitignore`)
2. **Chunks** code intelligently (syntax-aware for Python, JS, Go, Rust, Java, C/C++)
3. **Embeds** chunks as vectors using ML models
4. **Stores** in vector database (ChromaDB or Qdrant)
5. **Searches** semantically when you query

Pluggable architecture - swap databases, embedding models, or add new ones.

## API Usage

Use programmatically:

```python
from code_rag.api import CodeRAGAPI

api = CodeRAGAPI(database_type="chroma", embedding_model="all-MiniLM-L6-v2")
api.initialize_collection("myproject")

# Index
chunks = api.index_codebase("/path/to/project")

# Search
results = api.search("authentication logic", n_results=5)
for r in results:
    print(f"{r['file_path']} - {r['similarity']:.2f}")
```

## Documentation

- **[AGENTS.md](AGENTS.md)** - Developer onboarding and architecture overview
- **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** - Detailed implementation reference
- **[docs/mcp.md](docs/mcp.md)** - MCP server setup guide

## Supported Languages

Syntax-aware chunking for: Python, JavaScript, TypeScript, Go, Rust, Java, C, C++

Other languages use line-aware chunking (still works, just less context-aware).

## Requirements

- Python 3.8+
- Minimal dependencies (ChromaDB + sentence-transformers by default)
- Optional: OpenAI API key, Qdrant server

## Troubleshooting

**Import errors?** `pip install -e . --force-reinstall`

**Database issues?** `code-rag --reindex`

**Memory issues?** `export CODE_RAG_BATCH_SIZE="16"`

## Development

### Setup

```bash
# Install with dev dependencies
pip install -e ".[dev]"
```

### Testing & Linting

```bash
# Run tests
pytest

# Format code
black .
isort .

# Linting
flake8
```

## Contributing

1. Fork the repo
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

See [AGENTS.md](AGENTS.md) for architecture and [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for internals.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with [ChromaDB](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/), [sentence-transformers](https://www.sbert.net/), and [Tree-sitter](https://tree-sitter.github.io/)
