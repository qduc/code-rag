"""ChromaDB implementation of the database interface."""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings

from .database_interface import DatabaseInterface


class ChromaDatabase(DatabaseInterface):
    """ChromaDB implementation for vector storage."""

    def __init__(self, persist_directory: str = ".code-rag"):
        """
        Initialize ChromaDB client.

        Args:
            persist_directory: Directory to persist the database
        """
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = None

    def initialize(self, collection_name: str) -> None:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
        """
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add documents with embeddings to the database.

        Args:
            ids: Unique identifiers for the documents
            embeddings: Vector embeddings for the documents
            documents: The actual document contents
            metadatas: Optional metadata for each document
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

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
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        return results

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        return self.collection.count()

    def is_processed(self) -> bool:
        """
        Check if the codebase has already been processed.

        Returns:
            True if documents exist in the collection
        """
        return self.count() > 0

    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB PersistentClient handles persistence automatically
        pass
