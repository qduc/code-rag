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
  - File chunking with line-aware splitting to maintain code structure
  - Extensible file type detection (currently supports 40+ programming languages and formats)
- **Key Design**: Processes files → chunks each → yields chunk with metadata (file path, chunk position, total chunks) so results can be traced back to source

**When modifying**: File type detection and ignore patterns are configurable; chunking strategy is naive/character-based (v1), planned for syntax-aware improvements

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
- **Interface Contract**: Any embedding model must support single and batch embedding operations, and report its vector dimensionality
- **Default Implementation**: Local transformer-based model (lightweight, no API calls required)
- **Design Goal**: Extensible to support alternative providers (OpenAI, Hugging Face, etc.) without changing core logic
- **Key Pattern**: Embeddings are generated once during initial processing, then used for all queries; models are stateless

**When modifying**: New embedding implementations should follow the interface contract; embedding dimension must be consistent across all operations

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

## Developer Workflows

### Running the Tool
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

- **Chunking is currently naive**: V1 uses simple character-based splitting. Future versions will add syntax-aware chunking (AST-based, function-level, etc.) to respect code structure.
- **Graceful degradation in I/O**: File reading errors print a message but the pipeline continues processing other files. Gitignore parsing errors are silently ignored. Consider adding structured logging if this behavior needs improvement.
- **Multiple distance metrics**: Different database backends use different vector distance metrics (e.g., cosine vs. L2). Results may differ slightly between backends when querying.
- **Stateless embeddings**: Embedding models are initialized once and reused for all queries. They maintain no internal state about processed documents.
