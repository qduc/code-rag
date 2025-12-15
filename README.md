# Code-RAG

**Semantic code search for your entire codebase.** Ask questions in plain English, get relevant code snippets with source locations.

Instead of grepping for function names, ask "authentication logic" and find all related auth code across your project.

## Quick Start

```bash
# Create virtual environment, optional but recommended
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Index and search your codebase
code-rag

# That's it. Start asking questions about your code.
```

## Example

```
$ code-rag --path /home/user/myproject

Processing codebase... Found 247 files
Indexed 1,234 chunks in 12s

Query: authentication logic

Result 1 | Similarity: 0.85
File: src/auth/authenticator.py (lines 45-78)

class Authenticator:
    """Handle user authentication and session management."""

    def authenticate(self, username: str, password: str) -> bool:
        ...
```

## Why Use Code-RAG?

- **Understand unfamiliar codebases** - Ask questions instead of reading everything
- **Find examples** - "error handling with retries" finds all relevant patterns
- **Refactoring aid** - Locate all code related to a feature you're changing
- **Documentation** - Extract context for writing docs or onboarding

## Use with Claude Code (MCP Integration)

Code-RAG works as an MCP server, letting Claude automatically search your codebase during conversations.

```bash
# Register with Claude Code
claude mcp add code-rag --transport stdio path/to/venv/bin/code-rag-mcp
```

Then talk to Claude:
```
You: "Find the database connection logic"

Claude: [Automatically searches and finds the code]
        "I found the database connection logic in src/db/connection.py..."
```

See [MCP_SETUP.md](MCP_SETUP.md) for detailed setup.

## Basic Usage

```bash
# Different codebase
code-rag --path /path/to/repo

# Force reindex
code-rag --reindex

# More results
code-rag --results 10

# Different embedding model (better for code)
code-rag --model text-embedding-3-small # need to set OPENAI_API_KEY env

# Use Qdrant instead of ChromaDB
code-rag --database qdrant
```

## Configuration

Set via environment variables:

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

# Add custom ignore patterns (comma-separated)
export CODE_RAG_ADDITIONAL_IGNORE_PATTERNS="*.tmp,*.bak,logs/"
```

Full configuration options in [IMPLEMENTATION.md](IMPLEMENTATION.md#configuration-system).

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
from src.api import CodeRAGAPI

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
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)** - Detailed implementation reference
- **[MCP_SETUP.md](MCP_SETUP.md)** - MCP server setup guide

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

## Contributing

1. Fork the repo
2. Create feature branch
3. Make changes
4. Add tests
5. Submit PR

See [AGENTS.md](AGENTS.md) for architecture and [IMPLEMENTATION.md](IMPLEMENTATION.md) for internals.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built with [ChromaDB](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/), [sentence-transformers](https://www.sbert.net/), and [Tree-sitter](https://tree-sitter.github.io/)
