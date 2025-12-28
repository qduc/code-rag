"""Configuration management for the code-rag tool."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(env_path)

class Config:
    """Configuration handler for the code-rag application."""

    def __init__(self):
        """Initialize configuration from environment variables or defaults."""
        self.embedding_model = os.getenv(
            "CODE_RAG_EMBEDDING_MODEL", "nomic-ai/CodeRankEmbed"
        )
        self.database_type = os.getenv("CODE_RAG_DATABASE_TYPE", "chroma")

        # Use cache directory if not explicitly set
        if "CODE_RAG_DATABASE_PATH" in os.environ:
            self.database_path = os.getenv("CODE_RAG_DATABASE_PATH")
        else:
            self.database_path = self._get_default_database_path()

        self.chunk_size = self._get_int_env("CODE_RAG_CHUNK_SIZE", 1024)
        self.batch_size = max(1, self._get_int_env("CODE_RAG_BATCH_SIZE", 32))

        # Chunking configuration
        self.include_file_header = os.getenv("CODE_RAG_INCLUDE_FILE_HEADER", "true").lower() in ("true", "1", "yes")
        self.exclude_tests = os.getenv("CODE_RAG_EXCLUDE_TESTS", "false").lower() in ("true", "1", "yes")

        # Additional ignore patterns (comma-separated)
        additional_ignore_str = os.getenv("CODE_RAG_ADDITIONAL_IGNORE_PATTERNS", "")
        self.additional_ignore_patterns = [
            pattern.strip() for pattern in additional_ignore_str.split(",") if pattern.strip()
        ]

        self._sanitize_chunk_defaults()

        # Reranker configuration
        self.reranker_enabled = os.getenv("CODE_RAG_RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
        self.reranker_model = os.getenv("CODE_RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker_multiplier = int(os.getenv("CODE_RAG_RERANKER_MULTIPLIER", "2"))

        # Logging configuration
        self.log_level = os.getenv("CODE_RAG_LOG_LEVEL", "INFO").upper()

        # Incremental indexing configuration
        self.reindex_debounce_minutes = self._get_int_env("CODE_RAG_REINDEX_DEBOUNCE_MINUTES", 10)
        self.verify_changes_with_hash = os.getenv("CODE_RAG_VERIFY_CHANGES_WITH_HASH", "true").lower() in ("true", "1", "yes")

        # Model idle timeout configuration (in seconds)
        # Default: 1800 seconds (30 minutes)
        self.model_idle_timeout = self._get_int_env("CODE_RAG_MODEL_IDLE_TIMEOUT", 1800)

        # Shared embedding server configuration
        # When enabled, MCP instances share a single model via HTTP server
        # Server auto-starts on demand and auto-terminates when idle
        self.shared_server_enabled = os.getenv("CODE_RAG_SHARED_SERVER", "true").lower() in ("true", "1", "yes")
        self.shared_server_port = self._get_int_env("CODE_RAG_SHARED_SERVER_PORT", 8199)

    @staticmethod
    def _get_default_database_path() -> str:
        """Get the default database path in the user's cache directory."""
        # Use platformdirs logic for cross-platform cache directory
        if os.name == "nt":  # Windows
            cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        elif os.uname().sysname == "Darwin":  # macOS
            cache_dir = Path.home() / "Library" / "Caches"
        else:  # Linux and other Unix-like
            cache_dir = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))

        # Create code-rag subdirectory
        code_rag_cache = cache_dir / "code-rag"
        code_rag_cache.mkdir(parents=True, exist_ok=True)

        return str(code_rag_cache)

    def get_embedding_model(self) -> str:
        """Get the configured embedding model."""
        return self.embedding_model

    def get_database_type(self) -> str:
        """Get the configured database type."""
        return self.database_type

    def get_database_path(self) -> str:
        """Get the configured database path."""
        return self.database_path

    def get_chunk_size(self) -> int:
        """Get the configured chunk size."""
        return self.chunk_size

    def get_batch_size(self) -> int:
        """Get the configured batch size."""
        return self.batch_size

    def is_reranker_enabled(self) -> bool:
        """Get whether reranking is enabled."""
        return self.reranker_enabled

    def get_reranker_model(self) -> str:
        """Get the configured reranker model."""
        return self.reranker_model

    def get_reranker_multiplier(self) -> int:
        """Get the retrieval multiplier for reranking."""
        return self.reranker_multiplier

    def should_include_file_header(self) -> bool:
        """Get whether to include file headers in chunks."""
        return self.include_file_header

    def should_exclude_tests(self) -> bool:
        """Get whether to exclude test files from indexing."""
        return self.exclude_tests

    def get_additional_ignore_patterns(self) -> list[str]:
        """Get the additional ignore patterns."""
        return self.additional_ignore_patterns

    def get_log_level(self) -> str:
        """Get the configured log level."""
        return self.log_level

    def _get_int_env(self, name: str, default: int) -> int:
        """Parse integer env vars with a safe fallback."""
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def get_reindex_debounce_minutes(self) -> int:
        """Get the debounce interval for automatic reindexing in minutes."""
        return self.reindex_debounce_minutes

    def should_verify_changes_with_hash(self) -> bool:
        """Get whether to verify file changes with content hash."""
        return self.verify_changes_with_hash

    def get_model_idle_timeout(self) -> int:
        """Get the model idle timeout in seconds (default: 1800 = 30 minutes)."""
        return self.model_idle_timeout

    def is_shared_server_enabled(self) -> bool:
        """Get whether shared embedding server mode is enabled."""
        return self.shared_server_enabled

    def get_shared_server_port(self) -> int:
        """Get the shared embedding server port."""
        return self.shared_server_port

    def _sanitize_chunk_defaults(self) -> None:
        """Ensure chunk defaults form a sane pair when env vars are absent or invalid."""
        if self.chunk_size < 1:
            self.chunk_size = 1
