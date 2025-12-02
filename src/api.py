"""Public API layer for Code-RAG functionality.

This module provides a clean, reusable API for code-rag operations
that can be used by MCP servers, CLI tools, or other integrations.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Callable
from .config.config import Config
from .database.chroma_database import ChromaDatabase
from .database.qdrant_database import QdrantDatabase
from .database.database_interface import DatabaseInterface
from .embeddings.embedding_interface import EmbeddingInterface
from .embeddings.openai_embedding import OpenAIEmbedding
from .embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from .processor.file_processor import FileProcessor
from .reranker.reranker_interface import RerankerInterface
from .reranker.cross_encoder_reranker import CrossEncoderReranker


def generate_collection_name(codebase_path: str) -> str:
    """Generate a unique collection name for a codebase path.

    Uses a hash of the absolute path to ensure uniqueness while keeping
    collection names valid and manageable.

    Args:
        codebase_path: Path to the codebase

    Returns:
        A unique collection name string
    """
    # Resolve to absolute path for consistency
    abs_path = str(Path(codebase_path).resolve())

    # Create a short hash of the path
    path_hash = hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    # Collection name format: codebase_<hash>
    return f"codebase_{path_hash}"


def looks_like_codebase(root_path: Path, processor: FileProcessor) -> bool:
    """Heuristic: return True if the folder likely contains a codebase.

    Checks for common repo files/dirs, then falls back to counting discovered source files.
    """
    markers = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "Cargo.toml",
        "go.mod",
        ".git",
        "src",
    ]

    try:
        for m in markers:
            if (root_path / m).exists():
                return True

        # Fallback: if discover_files finds a reasonable number of source files,
        # consider it a codebase. Small threshold avoids misdetecting home dirs.
        files = processor.discover_files(str(root_path))
        return len(files) >= 5
    except Exception:
        # On any unexpected error, be conservative and treat as non-codebase
        return False


class CodeRAGAPI:
    """High-level API for Code-RAG operations."""

    def __init__(
        self,
        database_type: str = "chroma",
        database_path: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_enabled: bool = True,
        reranker_model: Optional[str] = None,
        reranker_multiplier: int = 2,
    ):
        """
        Initialize the Code-RAG API.

        Args:
            database_type: Type of database to use ("chroma" or "qdrant")
            database_path: Path to database storage (uses default cache if None)
            embedding_model: Name of the embedding model to use
            reranker_enabled: Whether to enable semantic reranking
            reranker_model: Name of the reranker model (uses default if None)
            reranker_multiplier: Retrieval multiplier for reranking (default: 2)
        """
        self.config = Config()
        self.database_type = database_type
        self.embedding_model_name = embedding_model
        self.reranker_enabled = reranker_enabled
        self.reranker_multiplier = reranker_multiplier

        # Use provided path or fall back to config default
        self.database_path = database_path or self.config.get_database_path()

        # Session state for tracking indexed codebases
        self._indexed_paths: Set[str] = set()
        # Track the currently active collection name
        self._active_collection: Optional[str] = None

        # Initialize embedding model
        self.embedding_model = self._create_embedding_model(embedding_model)

        # Initialize database
        self.database = self._create_database(database_type, self.database_path)

        # Initialize reranker if enabled
        self.reranker: Optional[RerankerInterface] = None
        if reranker_enabled:
            try:
                model_name = reranker_model or self.config.get_reranker_model()
                self.reranker = CrossEncoderReranker(model_name)
            except Exception as e:
                print(f"Warning: Failed to load reranker ({e}), disabling reranking")
                self.reranker = None

    def _create_embedding_model(self, model_name: str) -> EmbeddingInterface:
        """Create an embedding model instance based on the model name."""
        if model_name.startswith("text-embedding-"):
            return OpenAIEmbedding(model_name)
        else:
            return SentenceTransformerEmbedding(model_name)

    def _create_database(
        self, database_type: str, database_path: str
    ) -> DatabaseInterface:
        """Create a database instance based on the database type."""
        if database_type == "chroma":
            return ChromaDatabase(persist_directory=database_path)
        elif database_type == "qdrant":
            return QdrantDatabase(persist_directory=database_path)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")

    def initialize_collection(
        self, collection_name: str = "codebase", force_reindex: bool = False
    ) -> Optional[str]:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
            force_reindex: If True, delete existing collection before initializing

        Returns:
            None if initialization succeeded, or stored model name if dimension mismatch
        """
        if force_reindex:
            self.database.delete_collection(collection_name)

        vector_size = self.embedding_model.get_embedding_dimension()
        stored_model = self.database.initialize(
            collection_name, vector_size=vector_size, model_name=self.embedding_model_name
        )

        # If dimension mismatch, reload with the stored model
        if stored_model and stored_model != self.embedding_model_name:
            self.embedding_model = self._create_embedding_model(stored_model)
            self.embedding_model_name = stored_model
            return stored_model

        return None

    def index_codebase(
        self,
        codebase_path: str,
        collection_name: str = "codebase",
        progress_callback: Optional[callable] = None,
    ) -> int:
        """
        Index a codebase and store embeddings in the database.

        Args:
            codebase_path: Path to the codebase root directory
            collection_name: Name of the collection to store embeddings
            progress_callback: Optional callback for progress updates (file_count, total_files)

        Returns:
            Number of chunks processed

        Raises:
            ValueError: If path does not exist or is not a directory
        """
        path = Path(codebase_path).resolve()
        if not path.exists():
            raise ValueError(f"Path does not exist: {codebase_path}")
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {codebase_path}")

        processor = FileProcessor()

        # Discover files
        files = processor.discover_files(str(path))
        if not files:
            return 0

        total_chunks = 0
        batch_ids = []
        batch_contents = []
        batch_metadatas = []

        chunk_size = self.config.get_chunk_size()
        batch_size = self.config.get_batch_size()

        for i, file_path in enumerate(files):
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, len(files), file_path)

            # Process file into chunks
            chunks = processor.process_file(file_path, chunk_size)

            for chunk_data in chunks:
                batch_ids.append(chunk_data["id"])
                batch_contents.append(chunk_data["content"])
                batch_metadatas.append(chunk_data["metadata"])

                # When batch is full, embed and store
                if len(batch_ids) >= batch_size:
                    embeddings = self.embedding_model.embed_batch(batch_contents)
                    self.database.add(
                        ids=batch_ids,
                        embeddings=embeddings,
                        documents=batch_contents,
                        metadatas=batch_metadatas,
                    )
                    total_chunks += len(batch_ids)
                    batch_ids = []
                    batch_contents = []
                    batch_metadatas = []

        # Process remaining batch
        if batch_ids:
            embeddings = self.embedding_model.embed_batch(batch_contents)
            self.database.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_contents,
                metadatas=batch_metadatas,
            )
            total_chunks += len(batch_ids)

        return total_chunks

    def search(
        self,
        query: str,
        n_results: int = 5,
        collection_name: Optional[str] = None,
        expand_context: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search over the indexed codebase.

        Args:
            query: Natural language search query
            n_results: Number of results to return
            collection_name: Name of the collection to search (uses active collection if None)
            expand_context: If True, fetch adjacent chunks to provide more context

        Returns:
            List of search results, each containing:
                - content: The code chunk content
                - file_path: Path to the source file
                - chunk_index: Index of the chunk within the file
                - total_chunks: Total number of chunks in the file
                - start_line: Starting line number (if available)
                - end_line: Ending line number (if available)
                - function_name: Name of the enclosing function (if available)
                - class_name: Name of the enclosing class (if available)
                - symbol_type: Type of symbol (function, class, method)
                - similarity: Similarity score (0-1, higher is better)
        """
        # Use active collection if none specified
        if collection_name is None:
            if self._active_collection is None:
                raise ValueError("No collection specified and no active collection set. Call ensure_indexed() first.")
            collection_name = self._active_collection

        # Generate embedding for query
        query_embedding = self.embedding_model.embed(query)

        # Determine how many results to retrieve from database
        if self.reranker is not None:
            db_n_results = n_results * self.reranker_multiplier
        else:
            db_n_results = n_results

        # Search database
        results = self.database.query(query_embedding, n_results=db_n_results)

        if not results["documents"] or not results["documents"][0]:
            return []

        # Apply reranking if enabled
        if self.reranker is not None:
            try:
                # Extract documents from results
                documents = results["documents"][0]

                # Rerank documents
                reranked_indices = self.reranker.rerank(query, documents, top_k=n_results)

                # Reorder results based on reranking
                reranked_docs = []
                reranked_metadata = []
                reranked_scores = []

                for orig_idx, rerank_score in reranked_indices:
                    reranked_docs.append(results["documents"][0][orig_idx])
                    reranked_metadata.append(results["metadatas"][0][orig_idx])
                    reranked_scores.append(rerank_score)

                # Update results with reranked data
                results["documents"][0] = reranked_docs
                results["metadatas"][0] = reranked_metadata
                results["distances"][0] = reranked_scores

            except Exception as e:
                # Fall back to original results if reranking fails
                pass

        # Format results
        formatted_results = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Calculate similarity score
            if self.reranker is not None:
                similarity = distance  # Already a relevance score (higher = better)
            else:
                similarity = 1 - distance  # Convert cosine distance to similarity

            result = {
                "content": doc,
                "file_path": metadata.get("file_path", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0),
                "total_chunks": metadata.get("total_chunks", 1),
                "start_line": metadata.get("start_line"),
                "end_line": metadata.get("end_line"),
                "function_name": metadata.get("function_name"),
                "class_name": metadata.get("class_name"),
                "symbol_type": metadata.get("symbol_type"),
                "similarity": similarity,
            }
            formatted_results.append(result)

        # Expand context if requested
        if expand_context and formatted_results:
            formatted_results = self._expand_context(formatted_results)

        return formatted_results

    def _expand_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand search results by fetching adjacent chunks for more context.

        For each result, attempts to fetch the previous and next chunks from the same file
        and merges them into expanded_content.
        """
        expanded = []

        for result in results:
            file_path = result["file_path"]
            chunk_index = result["chunk_index"]
            total_chunks = result["total_chunks"]

            # Collect adjacent chunk indices
            adjacent_indices = []
            if chunk_index > 0:
                adjacent_indices.append(chunk_index - 1)
            adjacent_indices.append(chunk_index)  # Current chunk
            if chunk_index < total_chunks - 1:
                adjacent_indices.append(chunk_index + 1)

            # Try to fetch adjacent chunks
            chunks_content = {}
            chunks_metadata = {}

            for idx in adjacent_indices:
                chunk = self.get_chunk(file_path, idx)
                if chunk:
                    chunks_content[idx] = chunk["content"]
                    chunks_metadata[idx] = chunk

            # Build expanded content
            if len(chunks_content) > 1:
                # Sort by chunk index and concatenate
                sorted_indices = sorted(chunks_content.keys())
                expanded_content = "\n".join(chunks_content[i] for i in sorted_indices)

                # Update line numbers to span the expanded range
                first_chunk = chunks_metadata.get(sorted_indices[0], {})
                last_chunk = chunks_metadata.get(sorted_indices[-1], {})

                result["expanded_content"] = expanded_content
                result["expanded_start_line"] = first_chunk.get("start_line", result["start_line"])
                result["expanded_end_line"] = last_chunk.get("end_line", result["end_line"])
            else:
                # No expansion possible
                result["expanded_content"] = result["content"]
                result["expanded_start_line"] = result["start_line"]
                result["expanded_end_line"] = result["end_line"]

            expanded.append(result)

        return expanded

    def get_chunk(
        self, file_path: str, chunk_index: int, collection_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chunk by file path and chunk index.

        Args:
            file_path: Path to the source file
            chunk_index: Index of the chunk within the file
            collection_name: Name of the collection to search (uses active collection if None)

        Returns:
            Chunk data if found, None otherwise
        """
        # Use active collection if none specified
        if collection_name is None:
            if self._active_collection is None:
                raise ValueError("No collection specified and no active collection set. Call ensure_indexed() first.")
            collection_name = self._active_collection

        # Query using file path and chunk index in metadata
        # This is a simple implementation - could be optimized with metadata filtering
        query_embedding = self.embedding_model.embed(f"file:{file_path}")
        results = self.database.query(query_embedding, n_results=100)

        if not results["documents"] or not results["documents"][0]:
            return None

        # Find the matching chunk
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            if (
                metadata.get("file_path") == file_path
                and metadata.get("chunk_index") == chunk_index
            ):
                return {
                    "content": doc,
                    "file_path": metadata.get("file_path", "Unknown"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "start_line": metadata.get("start_line"),
                    "end_line": metadata.get("end_line"),
                }

        return None

    def is_processed(self) -> bool:
        """
        Check if the codebase has been processed.

        Returns:
            True if the database contains documents, False otherwise
        """
        try:
            return self.database.is_processed()
        except:
            return False

    def count(self) -> int:
        """
        Get the number of chunks in the database.

        Returns:
            Number of chunks stored
        """
        try:
            return self.database.count()
        except:
            return 0

    def ensure_indexed(
        self,
        codebase_path: str,
        collection_name: Optional[str] = None,
        force_reindex: bool = False,
        validate_codebase: bool = False,
        validation_callback: Optional[Callable[[Path], bool]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Ensure a codebase is indexed, handling all the logic from both CLI and MCP.

        This unified method merges the best practices from both implementations:
        - Session-based tracking (from MCP)
        - Codebase validation heuristic (from CLI)
        - Dimension mismatch handling (from CLI)
        - User confirmation for non-codebases (from CLI, via callback)
        - Auto-generates unique collection names per codebase path

        Args:
            codebase_path: Path to the codebase root directory
            collection_name: Name of the collection to use (auto-generated from path if None)
            force_reindex: If True, delete existing index and reindex
            validate_codebase: If True, check if path looks like a codebase
            validation_callback: Optional callback(path) -> bool for user confirmation
            progress_callback: Optional callback for progress updates

        Returns:
            Dict with keys:
                - success: bool - Whether operation succeeded
                - error: Optional[str] - Error message if failed
                - total_chunks: Optional[int] - Number of chunks indexed
                - already_indexed: bool - Whether codebase was already indexed
                - reloaded_model: Optional[str] - Model name if dimension mismatch occurred
        """
        # Validate path
        path = Path(codebase_path).resolve()
        if not path.exists():
            return {"success": False, "error": f"Path does not exist: {codebase_path}"}
        if not path.is_dir():
            return {"success": False, "error": f"Path is not a directory: {codebase_path}"}

        path_str = str(path)

        # Auto-generate collection name from path if not provided
        if collection_name is None:
            collection_name = generate_collection_name(path_str)

        # Check session cache first (fast path)
        if not force_reindex and path_str in self._indexed_paths:
            return {"success": True, "already_indexed": True, "total_chunks": self.count()}

        # Try to find existing data in the new collection first
        # Initialize collection first (needed to check if data exists)
        # This loads existing data from disk if available
        stored_model = self.initialize_collection(collection_name, force_reindex=force_reindex)
        reloaded_model = None

        if stored_model and stored_model != self.embedding_model_name:
            # Dimension mismatch - reload with stored model
            self.embedding_model = self._create_embedding_model(stored_model)
            self.embedding_model_name = stored_model
            reloaded_model = stored_model

        # Set this as the active collection for subsequent operations
        self._active_collection = collection_name

        # Now that collection is initialized, check if database already has data
        if not force_reindex and self.is_processed() and self.count() > 0:
            # Mark as indexed and return
            self._indexed_paths.add(path_str)
            return {"success": True, "already_indexed": True, "total_chunks": self.count(), "reloaded_model": reloaded_model}

        # Need to index - perform validation if requested
        if validate_codebase:
            processor = FileProcessor()
            if not looks_like_codebase(path, processor):
                # Ask for confirmation via callback
                if validation_callback:
                    if not validation_callback(path):
                        return {
                            "success": False,
                            "error": "User declined to index non-codebase directory",
                        }
                # If no callback, proceed anyway (MCP mode)

        # Index the codebase
        try:
            total_chunks = self.index_codebase(
                path_str,
                collection_name=collection_name,
                progress_callback=progress_callback,
            )

            # Mark as indexed
            self._indexed_paths.add(path_str)

            return {
                "success": True,
                "already_indexed": False,
                "total_chunks": total_chunks,
                "reloaded_model": reloaded_model,
            }

        except Exception as e:
            return {"success": False, "error": f"Indexing failed: {str(e)}"}

    def close(self):
        """Close database connections and cleanup resources."""
        if self.database:
            self.database.close()
