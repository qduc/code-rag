"""Configuration management for the code-rag tool."""

import os
from typing import Optional


class Config:
    """Configuration handler for the code-rag application."""

    def __init__(self):
        """Initialize configuration from environment variables or defaults."""
        self.embedding_model = os.getenv(
            "CODE_RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.database_type = os.getenv("CODE_RAG_DATABASE_TYPE", "chroma")
        self.database_path = os.getenv("CODE_RAG_DATABASE_PATH", ".code-rag")
        self.chunk_size = int(os.getenv("CODE_RAG_CHUNK_SIZE", "1024"))
        self.batch_size = int(os.getenv("CODE_RAG_BATCH_SIZE", "32"))

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
