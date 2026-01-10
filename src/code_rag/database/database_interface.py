"""Abstract base class for vector database."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class DatabaseInterface(ABC):
    """Abstract interface for vector databases."""

    @abstractmethod
    def initialize(
        self,
        collection_name: str,
        vector_size: int = 384,
        model_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
            vector_size: Dimension of the embedding vectors (default: 384 for all-MiniLM-L6-v2)
            model_name: Name of the embedding model used to create the vectors

        Returns:
            None if initialization succeeded with the requested parameters,
            or the stored model name if there's a dimension mismatch (caller should
            reload with this model).
        """
        pass

    @abstractmethod
    def get_model_name(self) -> Optional[str]:
        """
        Get the model name stored in the collection metadata.

        Returns:
            The model name if stored, None otherwise
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
    def query(self, embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
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
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        pass

    @abstractmethod
    def get_all_ids(self) -> List[str]:
        """
        Get all document IDs in the collection.

        Returns:
            List of all document IDs
        """
        pass

    @abstractmethod
    def get_ids_by_file(self, file_path: str) -> List[str]:
        """
        Get all document IDs for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for that file
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
