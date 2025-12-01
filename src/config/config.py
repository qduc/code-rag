"""Configuration management for the code-rag tool."""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Configuration handler for the code-rag application."""

    def __init__(self):
        """Initialize configuration from environment variables or defaults."""
        self.embedding_model = os.getenv(
            "CODE_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.database_type = os.getenv("CODE_RAG_DATABASE_TYPE", "chroma")

        # Use cache directory if not explicitly set
        if "CODE_RAG_DATABASE_PATH" in os.environ:
            self.database_path = os.getenv("CODE_RAG_DATABASE_PATH")
        else:
            self.database_path = self._get_default_database_path()

        self.chunk_size = int(os.getenv("CODE_RAG_CHUNK_SIZE", "1024"))
        self.batch_size = int(os.getenv("CODE_RAG_BATCH_SIZE", "32"))

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
