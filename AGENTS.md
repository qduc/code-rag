# Code-RAG: AI Coding Agent Instructions

## Project Overview

**Code-RAG** is a CLI tool that converts source codebases into searchable vector embeddings using RAG (Retrieval-Augmented Generation). Users run the tool on a codebase root directory, which processes files into embeddings and stores them in a vector database, then provides an interactive query session for semantic code search.

### Architecture Highlights

- **Plugin Pattern**: Pluggable database and embedding implementations via abstract base classes (`DatabaseInterface`, `EmbeddingInterface`)
- **Default Stack**: ChromaDB (vector store) + sentence-transformers (embeddings)
- **Processing Pipeline**: File discovery → Read → Chunk → Embed → Store → Query
- **CLI Entry Point**: Installed as `code-rag` command via setuptools entry points

## Key Components

### 1. **File Processor**
- **Purpose**: Discovers source files, reads them with flexible encoding, and chunks them for embedding
- **Core Responsibilities**:
  - File discovery via recursive traversal, respecting `.gitignore` and standard ignore patterns (e.g., `node_modules`, `.git`, build directories)
  - File reading with multiple encoding fallbacks to handle diverse source files
  - File chunking using syntax-aware splitting for supported languages, with fallback to line-aware chunking
  - Extensible file type detection (currently supports 40+ programming languages and formats)
- **Key Design**: Processes files → chunks each using syntax trees when available → yields chunk with metadata (file path, chunk position, total chunks) so results can be traced back to source

**When modifying**: File type detection and ignore patterns are configurable; syntax-aware chunking uses tree-sitter for supported languages (Python, JavaScript, TypeScript, Go, Rust, Java, C++, C) with automatic fallback to character-based line-aware chunking for unsupported languages

### 2. **Database Layer**
- **Architecture**: Abstract interface with multiple pluggable implementations
- **Core Responsibility**: Store processed chunks with vectors, support similarity search, track processing state
- **Interface Contract**: Any database must support initialization, batch document insertion with embeddings, vector similarity queries, and resource cleanup
- **Default Implementation** (ChromaDB): Lightweight embedded database requiring no server setup
- **Alternative Implementation** (Qdrant): Network-based option for distributed deployments
- **Key Pattern**: Implementations are interchangeable via configuration; adding a new database means extending the interface and wiring it into initialization

**When modifying**: Prefer adding new database implementations rather than changing the interface; maintain the contract that databases can report processing state via count/existence checks

### 3. **Embedding Layer**
- **Architecture**: Abstract interface with multiple pluggable implementations
- **Core Responsibility**: Convert text into vector embeddings for similarity search
- **Interface Contract**: Any embedding model must support single and batch embedding operations, report its vector dimensionality, and optionally support query-specific prefixes via `embed_query()`
- **Default Implementation**: Local transformer-based model (lightweight, no API calls required)
- **Supported Models**:
  - `sentence-transformers/all-MiniLM-L6-v2`: Default general-purpose embedding model
  - `nomic-ai/CodeRankEmbed`: Code-optimized model with automatic query instruction prefixing
  - `text-embedding-3-small`: OpenAI embedding model (requires API key)
- **Design Goal**: Extensible to support alternative providers (OpenAI, Hugging Face, etc.) without changing core logic
- **Key Pattern**:
  - Code chunks are embedded once during indexing using `embed()` or `embed_batch()`
  - User queries are embedded using `embed_query()`, which automatically prepends model-specific instruction prefixes when needed (e.g., "Represent this query for searching relevant code: " for CodeRankEmbed)
  - Models are stateless and reused across all operations

**When modifying**:
- New embedding implementations should follow the interface contract; embedding dimension must be consistent across all operations
- Models requiring special query instructions should:
  1. Define instruction prefix in `QUERY_INSTRUCTION_PREFIX` class variable
  2. Override `embed_query()` to apply the prefix before embedding
  3. Keep `embed()` and `embed_batch()` unchanged for document embedding

### 4. **Configuration** (`src/config/config.py`)
- Reads from environment variables with defaults:
  - `CODE_RAG_EMBEDDING_MODEL`: Model name (default: "all-MiniLM-L6-v2")
  - `CODE_RAG_DATABASE_TYPE`: "chroma" or "qdrant" (default: "chroma")
  - `CODE_RAG_DATABASE_PATH`: Persist directory (default: ".code-rag")
  - `CODE_RAG_CHUNK_SIZE`: Characters per chunk (default: 1024)
  - `CODE_RAG_BATCH_SIZE`: Docs to batch before insert (default: 32)

### 5. **Main CLI**
- **Role**: Orchestrates the entire pipeline—coordinates file processing, database selection, embedding generation, and query sessions
- **Responsibilities**:
  1. Parse command-line arguments and load configuration
  2. Initialize selected database and embedding implementations
  3. Check if codebase has been processed; run processing if needed or if `--reindex` flag is set
  4. Start interactive query loop where user queries are embedded and searched
- **Data Flow**: Configuration (env vars / args) → File discovery → Batch processing → Database storage → Query session
- **Key Insight**: The CLI is the only place where database and embedding implementations are instantiated; this is where pluggable behavior is wired in

**When modifying**: Database/embedding initialization is centralized; new implementations require conditional instantiation logic here

### 6. **MCP Server** (`src/mcp_server.py`)
- **Purpose**: Exposes Code-RAG functionality as an MCP (Model Context Protocol) server that Claude and other AI assistants can connect to
- **Core Responsibilities**:
  - Implements MCP protocol via stdio transport for client-server communication
  - Provides `search_codebase` tool for semantic code search
  - Auto-indexes codebases transparently on first search (no user confirmation needed)
  - Formats and returns search results with file paths, line numbers, and relevance scores
- **Architecture**:
  - Uses the CodeRAG API internally (shares same backend as CLI)
  - Async-based using asyncio for non-blocking I/O
  - Entry point: `code-rag-mcp` command (synchronous wrapper around async server)
- **Key Design**:
  - **Auto-indexing**: Automatically indexes codebases on first search without user intervention
  - **Transparent**: Hides implementation details (collections, chunking strategy) from the AI client
  - **Simple**: Single tool that "just works" - search is the only exposed operation
  - **Silent progress**: No progress callbacks or validation prompts in MCP mode
- **Entry Point Pattern**: The `main()` function is a synchronous wrapper that calls `asyncio.run(async_main())`, required for setuptools console_scripts compatibility

**When modifying**:
- The entry point MUST be synchronous (`def main()`) not async, as setuptools calls it directly
- MCP tool schema changes require updating both the `list_tools()` handler and `call_tool()` implementation
- Auto-indexing behavior is controlled via `validate_codebase=False` parameter to skip user prompts

## Developer Workflows

### Running the Tool

#### CLI Mode
```bash
# Install in development mode
pip install -e .

# Basic usage (processes current directory, default ChromaDB)
code-rag

# Process specific codebase with Qdrant
code-rag --path /path/to/repo --database qdrant

# Force reprocessing
code-rag --reindex

# Use different embedding model
code-rag --model sentence-transformers/paraphrase-MiniLM-L6-v2
```

#### MCP Server Mode
```bash
# Run the MCP server (typically called by MCP clients like Claude Desktop)
code-rag-mcp

# The server communicates via stdio and waits for JSON-RPC messages
# Configure in your MCP client (e.g., Claude Desktop config):
# {
#   "mcpServers": {
#     "code-rag": {
#       "command": "/path/to/venv/bin/code-rag-mcp"
#     }
#   }
# }
```

### Testing
```bash
# Run tests (pytest)
pytest

# Code quality
black src/
flake8 src/
```

### Common Modifications
- **Change default database**: Modify the initialization logic in the CLI to select a different database when none is specified
- **Add new file extension**: Extend file type detection by adding patterns or extensions to the file processor's type registry
- **Adjust chunking parameters**: Modify the chunking algorithm in the file processor (currently character-based with line-break awareness)
- **New database**: Create a new database implementation class extending the database interface, then wire it into the CLI's initialization logic
- **New embedding provider**: Create a new embedding implementation class extending the embedding interface, then wire it into the CLI's initialization logic
- **Add MCP tools**: Define new tools in `list_tools()` and implement handlers in `call_tool()` in the MCP server
- **IMPORTANT - MCP entry point**: If modifying async server code, remember the entry point pattern: `main()` must be synchronous and call `asyncio.run(async_main())`. Never make `main()` itself async, as setuptools cannot handle async entry points

## Project-Specific Patterns

### 1. **Batch Processing**
Chunks are embedded in batches (default batch size 32) before database insertion. This pattern optimizes embedding throughput and reduces database I/O overhead. Batches accumulate as files are processed, then flush to the database when reaching the batch size threshold or when processing completes.

### 2. **Metadata Tracking**
Each chunk carries metadata describing its source: file path, position within the file (chunk index), and total number of chunks from that file. This enables search results to be traced back to their exact source location.

### 3. **Idempotency via State Checking**
Before processing, the tool queries the database to check if documents already exist (via `is_processed()` or count). If documents are found and `--reindex` is not set, processing is skipped. This prevents duplicate embeddings from multiple runs on the same codebase.

### 4. **Plugin Architecture**
New implementations (database, embedding) must adhere to a pattern:
- Inherit from corresponding abstract interface (defined in dedicated interface files)
- Implement all interface methods (interface enforces this via ABC)
- Be instantiated conditionally in the CLI orchestration layer (e.g., based on configuration)
- Maintain type consistency with their interface contract

### 5. **Config via Environment**
Configuration defaults come from environment variables, allowing deployment flexibility (Docker, CI/CD, serverless). Each configuration option maps to an environment variable with a hardcoded fallback default. CLI arguments override environment settings.

## Integration Points & Dependencies

- **External**: chromadb, qdrant-client, sentence-transformers, python-dotenv
- **Cross-Module**:
  - CLI orchestrates by importing the file processor, all database implementations, all embedding implementations, and configuration
  - Database implementations are independent of each other (pure plugin pattern)
  - Embedding implementations are independent and self-contained
- **Key Principle**: No circular imports; each module imports only what it directly uses

## Notes for Contributors

- **Syntax-aware chunking**: The processor uses tree-sitter to parse supported languages and chunk by syntax boundaries (functions, classes, etc.). When tree-sitter parsing fails or the language is unsupported, it automatically falls back to character-based line-aware chunking.
- **Supported syntax languages**: Python, JavaScript, TypeScript, Go, Rust, Java, C++, C. Additional languages can be added by extending `SyntaxChunker.LANGUAGE_PACKAGES` and installing the corresponding tree-sitter binding.
- **Graceful degradation in I/O**: File reading errors print a message but the pipeline continues processing other files. Gitignore parsing errors are silently ignored. Similarly, syntax chunking errors fall back to basic chunking. Consider adding structured logging if this behavior needs improvement.
- **Multiple distance metrics**: Different database backends use different vector distance metrics (e.g., cosine vs. L2). Results may differ slightly between backends when querying.
- **Stateless embeddings**: Embedding models are initialized once and reused for all queries. They maintain no internal state about processed documents.
