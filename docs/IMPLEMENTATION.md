# Code-RAG: Implementation Reference

This document contains detailed implementation notes, patterns, and design decisions. Read this when you're working on specific components or need to understand internal behavior.

## Table of Contents

1. [File Processor Details](#file-processor-details)
2. [Database Layer Implementation](#database-layer-implementation)
3. [Embedding Layer Implementation](#embedding-layer-implementation)
4. [Configuration System](#configuration-system)
5. [MCP Server Architecture](#mcp-server-architecture)
6. [Processing Patterns](#processing-patterns)
7. [Testing Strategy](#testing-strategy)
8. [Dependencies](#dependencies)

---

## File Processor Details

### Responsibilities
- File discovery via recursive traversal
- Respects `.gitignore`, standard ignore patterns (e.g., `node_modules`, `.git`, build directories), and configurable additional patterns
- File reading with multiple encoding fallbacks to handle diverse source files
- Syntax-aware chunking for supported languages, with fallback to line-aware chunking
- Extensible file type detection (40+ programming languages and formats)

### Chunking Strategy

**Syntax-aware chunking** (preferred):
- Uses tree-sitter to parse supported languages: Python, JavaScript, TypeScript, Go, Rust, Java, C++, C
- Chunks by syntax boundaries (functions, classes, etc.)
- Automatically falls back to line-aware chunking on parse errors

**Line-aware chunking** (fallback):
- Character-based with line-break awareness
- Ensures chunks don't split mid-line
- Used for unsupported languages or when syntax parsing fails

### Metadata Format

Each chunk includes:
```python
{
    "file_path": "src/code_rag/example.py",
    "chunk_index": 0,           # Position in file (0-indexed)
    "total_chunks": 5,          # Total chunks from this file
    "content": "chunk text..."  # The actual code chunk
}
```

### Adding New Language Support

Extend `SyntaxChunker.LANGUAGE_PACKAGES` and install tree-sitter binding:
```python
LANGUAGE_PACKAGES = {
    "python": "tree-sitter-python",
    "javascript": "tree-sitter-javascript",
    # Add new languages here
}
```

### Error Handling

**File reading errors**: Print message, continue processing other files
**Gitignore parsing errors**: Silently ignored
**Syntax chunking errors**: Automatically fall back to basic chunking

Consider adding structured logging if more visibility is needed.

---

## Database Layer Implementation

### Interface Contract

Any database implementation must provide:

```python
class DatabaseInterface(ABC):
    @abstractmethod
    def initialize(self, collection_name: str, dimension: int) -> None:
        """Initialize/connect to database with collection and vector dimension"""
        pass

    @abstractmethod
    def add_documents(self, documents: List[str], embeddings: List[List[float]],
                     metadatas: List[dict]) -> None:
        """Batch insert documents with their embeddings and metadata"""
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], n_results: int) -> List[dict]:
        """Similarity search, return top n results"""
        pass

    @abstractmethod
    def is_processed(self, collection_name: str) -> bool:
        """Check if collection has documents (for idempotency)"""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources"""
        pass
```

### ChromaDB Implementation

**Properties**:
- Embedded database (no server required)
- Persistent on-disk storage
- Cosine similarity for vector search
- Collection-based organization

**Initialization**: Creates/loads collection with specified dimension

**Key behavior**: Uses `cosine` distance metric by default

### Qdrant Implementation

**Properties**:
- Network-based (requires Qdrant server)
- Supports distributed deployments
- L2 distance for vector search
- Collection-based organization

**Initialization**: Connects to server, creates collection if needed

**Key behavior**: Uses L2 (Euclidean) distance metric

### Distance Metrics Consideration

Different backends use different vector distance metrics:
- ChromaDB: Cosine similarity
- Qdrant: L2 distance

Results may differ slightly between backends when querying the same data. Both are valid; choice depends on use case and infrastructure requirements.

---

## Embedding Layer Implementation

### Interface Contract

```python
class EmbeddingInterface(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed single text string"""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts efficiently"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text (may apply model-specific prefixes)"""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Report vector dimension"""
        pass
```

### Supported Models

#### 1. `all-MiniLM-L6-v2` (Default)
- **Type**: General-purpose sentence transformer
- **Dimension**: 384
- **Best for**: Balanced performance and speed
- **Query prefix**: None
- **API key**: Not required

#### 2. `nomic-ai/CodeRankEmbed`
- **Type**: Code-optimized embedding model
- **Dimension**: 768
- **Best for**: Code semantic search
- **Query prefix**: "Represent this query for searching relevant code: "
- **API key**: Not required
- **Implementation note**: Overrides `embed_query()` to apply prefix

#### 3. `text-embedding-3-small` (OpenAI)
- **Type**: OpenAI API-based
- **Dimension**: 1536
- **Best for**: High quality, cloud-based
- **Query prefix**: None
- **API key**: Required (`OPENAI_API_KEY` environment variable)

### Query Instruction Prefix Pattern

Some models perform better when queries have instruction prefixes. Implement this pattern:

```python
class CodeRankEmbedding(EmbeddingInterface):
    QUERY_INSTRUCTION_PREFIX = "Represent this query for searching relevant code: "

    def embed_query(self, text: str) -> List[float]:
        """Override to prepend instruction prefix for queries"""
        prefixed_text = self.QUERY_INSTRUCTION_PREFIX + text
        return self.embed(prefixed_text)

    def embed(self, text: str) -> List[float]:
        """Regular embedding for documents (no prefix)"""
        # ... implementation

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embedding for documents (no prefix)"""
        # ... implementation
```

**Important**: Only apply prefix in `embed_query()`. Keep `embed()` and `embed_batch()` unchanged so document embeddings remain unprefixed.

### Stateless Design

Embedding models:
- Are initialized once at startup
- Maintain no internal state about processed documents
- Can be reused across all operations without side effects
- Thread-safe (for most implementations)

---

## Configuration System

### Environment Variables

Located in `src/code_rag/config/config.py`. All have fallback defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODE_RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `CODE_RAG_DATABASE_TYPE` | `chroma` | Database backend (`chroma` or `qdrant`) |
| `CODE_RAG_DATABASE_PATH` | `.code-rag` | Persist directory for embeddings |
| `CODE_RAG_CHUNK_SIZE` | `1024` | Characters per code chunk |
| `CODE_RAG_BATCH_SIZE` | `32` | Documents per batch before insert |
| `CODE_RAG_ADDITIONAL_IGNORE_PATTERNS` | `` | Comma-separated additional ignore patterns |

### CLI Override Pattern

CLI arguments override environment variables:
```bash
# Environment default: chroma
export CODE_RAG_DATABASE_TYPE=qdrant

# CLI override: uses chroma despite environment
code-rag --database chroma
```

### Adding New Configuration

1. Add to environment variable list in `config.py`
2. Provide sensible default
3. Add CLI argument in `main.py` if needed
4. Document in this file

---

## MCP Server Architecture

### Purpose

Exposes Code-RAG functionality to AI assistants (Claude, etc.) via Model Context Protocol.

### Communication Pattern

- **Transport**: stdio (standard input/output)
- **Protocol**: JSON-RPC messages
- **Tools**: Single tool `search_codebase`
- **Auto-indexing**: Transparently indexes codebases on first search

### Entry Point Pattern (Critical)

```python
async def async_main():
    """Async server implementation"""
    # ... MCP server logic

def main():
    """Synchronous entry point for setuptools"""
    asyncio.run(async_main())
```

**Why this matters**: Setuptools `console_scripts` cannot handle async entry points. The entry point MUST be synchronous (`def main()`) and call `asyncio.run()` internally.

**Never do this**:
```python
# WRONG - setuptools cannot call this
async def main():
    # ... async code
```

### Auto-indexing Behavior

MCP mode auto-indexes without user prompts:
- First search on a codebase triggers indexing
- No validation prompts or progress callbacks
- Silent operation for seamless AI assistant integration
- Controlled via `validate_codebase=False` parameter

### Adding New MCP Tools

1. Define tool schema in `list_tools()` handler:
```python
@server.list_tools()
async def list_tools() -> List[Dict]:
    return [
        {
            "name": "search_codebase",
            "description": "...",
            "inputSchema": {...}
        },
        # Add new tool here
    ]
```

2. Implement handler in `call_tool()`:
```python
@server.call_tool()
async def call_tool(name: str, arguments: Dict) -> List[Dict]:
    if name == "search_codebase":
        # ... existing implementation
    elif name == "new_tool":
        # ... new implementation
```

---

## Processing Patterns

### 1. Batch Processing

**Optimization**: Chunks are embedded and inserted in batches (default 32) rather than individually.

**Flow**:
1. Process files → accumulate chunks
2. When batch size reached → embed all chunks
3. Insert batch into database
4. Continue until all files processed
5. Flush remaining chunks

**Rationale**: Reduces database I/O overhead and improves embedding throughput (models process batches more efficiently).

### 2. Idempotency via State Checking

**Pattern**: Before processing, check if database already contains documents.

**Implementation**:
```python
if db.is_processed(collection_name) and not args.reindex:
    print("Codebase already processed. Use --reindex to force.")
    return
```

**Behavior**:
- No `--reindex` flag: Skip processing if documents exist
- With `--reindex` flag: Always reprocess and replace embeddings

**Rationale**: Prevents duplicate embeddings from multiple runs, saves time on subsequent invocations.

### 3. Metadata Tracking

Every chunk stores metadata:
- **file_path**: Source file location
- **chunk_index**: Position within file (0-indexed)
- **total_chunks**: How many chunks came from this file

**Purpose**: Search results can be traced back to exact source location. Enables features like "show me the full file" or "navigate to line X".

### 4. Plugin Architecture

**Design**: Database and embedding implementations are interchangeable via configuration.

**Wiring**:
- Interfaces defined in separate files (`database_interface.py`, `embedding_interface.py`)
- Implementations are independent modules
- CLI orchestration layer instantiates based on configuration:

```python
# Example from main.py
if args.database == "chroma":
    db = ChromaDB(persist_directory=args.path)
elif args.database == "qdrant":
    db = QdrantDB(url=args.qdrant_url)
```

**Adding new implementation**:
1. Create class extending interface
2. Implement all abstract methods
3. Add conditional instantiation in CLI
4. Wire configuration if needed

---

## Testing Strategy

### Running Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=code_rag

# Specific module
pytest tests/test_file_processor.py
```

### Test Structure

- **Unit tests**: Individual components (file processor, chunking, embeddings)
- **Integration tests**: Database operations, end-to-end indexing
- **Mock patterns**: External dependencies (OpenAI API, Qdrant server) are mocked

### Code Quality

```bash
# Format code
black src/code_rag/

# Lint
flake8 src/code_rag/

# Type checking (if configured)
mypy src/code_rag/
```

---

## Dependencies

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `chromadb` | Default vector database |
| `qdrant-client` | Alternative vector database |
| `sentence-transformers` | Local embedding models |
| `openai` | OpenAI API embeddings |
| `tree-sitter` | Syntax-aware chunking |
| `python-dotenv` | Environment configuration |
| `mcp` | Model Context Protocol server |

### Tree-sitter Language Bindings

Required for syntax-aware chunking:
- `tree-sitter-python`
- `tree-sitter-javascript`
- `tree-sitter-typescript`
- `tree-sitter-go`
- `tree-sitter-rust`
- `tree-sitter-java`
- `tree-sitter-cpp`
- `tree-sitter-c`

### Cross-Module Imports

**Principle**: No circular imports. Each module imports only what it directly uses.

**Flow**:
- CLI imports: file processor, database implementations, embedding implementations, config
- Database implementations: Independent of each other
- Embedding implementations: Independent and self-contained
- File processor: No dependencies on database or embedding layers

---

## Notes for Contributors

### Graceful Degradation

System continues processing even when individual operations fail:
- File reading errors → print message, continue with other files
- Gitignore parsing errors → silently ignored, continue
- Syntax chunking errors → fall back to basic chunking, continue

Consider adding structured logging if more visibility into failures is needed.

### Multiple Distance Metrics

Different backends use different similarity metrics:
- ChromaDB: Cosine similarity (normalized dot product)
- Qdrant: L2 distance (Euclidean distance)

Both are valid; results may differ slightly. This is expected behavior, not a bug.

### Adding Support for New Languages

1. Install tree-sitter binding: `pip install tree-sitter-{language}`
2. Add to `SyntaxChunker.LANGUAGE_PACKAGES` mapping
3. Test on sample files from that language
4. Update documentation

### Performance Considerations

- Batch size affects memory usage vs. throughput
- Larger chunks preserve more context but reduce granularity
- Syntax-aware chunking is slower but produces better results
- Embedding models vary significantly in speed and quality

Tune based on your codebase size and quality requirements.
