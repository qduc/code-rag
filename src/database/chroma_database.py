"""ChromaDB implementation of the database interface."""

import os
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

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

    def initialize(self, collection_name: str, vector_size: int = 384) -> None:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
            vector_size: Dimension of the embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        try:
            self.collection = self.client.get_collection(name=collection_name)

            # Check dimension compatibility
            # 1. Check explicit metadata
            existing_dim = self.collection.metadata.get("dimension")

            # 2. If no metadata, check existing data
            if existing_dim is None and self.collection.count() > 0:
                peek = self.collection.peek(limit=1)
                if peek["embeddings"]:
                    existing_dim = len(peek["embeddings"][0])

            if existing_dim is not None and int(existing_dim) != vector_size:
                print(f"Dimension mismatch: Collection '{collection_name}' has dimension {existing_dim}, requested {vector_size}.")
                print("Recreating collection with new dimension...")
                self.client.delete_collection(collection_name)
                self.collection = None

        except Exception:
            # Collection does not exist or other error occurred
            self.collection = None

        if self.collection is None:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "dimension": vector_size},
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

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection from the database.

        Args:
            collection_name: Name of the collection to delete
        """
        try:
            self.client.delete_collection(collection_name)
            if self.collection and self.collection.name == collection_name:
                self.collection = None
        except (ValueError, NotFoundError):
            # Collection does not exist, which is fine
            pass

    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB PersistentClient handles persistence automatically
        pass
