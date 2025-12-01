"""Abstract base class for vector database."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class DatabaseInterface(ABC):
    """Abstract interface for vector databases."""

    @abstractmethod
    def initialize(self, collection_name: str) -> None:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
        """
        pass

    @abstractmethod
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]] = None,
    ) -> None:
        """
        Add documents with embeddings to the database.

        Args:
            ids: Unique identifiers for the documents
            embeddings: Vector embeddings for the documents
            documents: The actual document contents
            metadatas: Optional metadata for each document
        """
        pass

    @abstractmethod
    def query(
        self, embedding: List[float], n_results: int = 5
    ) -> Dict[str, Any]:
        """
        Query the database with an embedding vector.

        Args:
            embedding: The query embedding vector
            n_results: Number of results to return

        Returns:
            Query results containing distances and documents
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
