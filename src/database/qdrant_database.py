"""Qdrant implementation of the database interface."""

import os
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from .database_interface import DatabaseInterface


class QdrantDatabase(DatabaseInterface):
    """Qdrant implementation for vector storage."""

    def __init__(self, persist_directory: str = ".code-rag", host: Optional[str] = None, port: Optional[int] = None):
        """
        Initialize Qdrant client.

        Args:
            persist_directory: Directory to persist the database (for local storage)
            host: Qdrant server host (for remote connection)
            port: Qdrant server port (for remote connection)
        """
        self.persist_directory = persist_directory

        # Initialize client for local or remote usage
        if host:
            self.client = QdrantClient(host=host, port=port or 6333)
        else:
            # Use local storage
            os.makedirs(persist_directory, exist_ok=True)
            self.client = QdrantClient(path=persist_directory)

        self.collection_name = None
        self.vector_size = None

    def initialize(self, collection_name: str, vector_size: int = 384) -> None:
        """
        Initialize or get a collection in the database.

        Args:
            collection_name: Name of the collection to initialize
            vector_size: Dimension of the embedding vectors (default: 384 for all-MiniLM-L6-v2)
        """
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_exists = any(col.name == collection_name for col in collections)

        if collection_exists:
            # Check for dimension mismatch
            collection_info = self.client.get_collection(collection_name)
            current_vectors = collection_info.config.params.vectors
            current_size = None

            # Handle VectorParams object or dict
            if hasattr(current_vectors, "size"):
                current_size = current_vectors.size
            elif isinstance(current_vectors, dict) and "size" in current_vectors:
                current_size = current_vectors["size"]

            if current_size is not None and current_size != vector_size:
                print(f"Dimension mismatch: Collection '{collection_name}' has size {current_size}, requested {vector_size}.")
                print("Recreating collection with new dimension...")
                self.client.delete_collection(collection_name)
                collection_exists = False

        if not collection_exists:
            # Create collection with cosine distance
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
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
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        # Prepare points for Qdrant
        points = []
        for i, (doc_id, embedding, document) in enumerate(zip(ids, embeddings, documents)):
            payload = {"document": document}
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upload points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
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
            Query results containing distances and documents (ChromaDB-compatible format)
        """
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=n_results,
        )

        # Convert to ChromaDB-compatible format
        documents = []
        metadatas = []
        distances = []
        ids = []

        for result in search_results:
            ids.append(result.id)
            documents.append(result.payload.get("document", ""))

            # Extract metadata (excluding the document field)
            metadata = {k: v for k, v in result.payload.items() if k != "document"}
            metadatas.append(metadata)

            # Qdrant returns similarity score, convert to distance
            # For cosine similarity: distance = 1 - score
            distances.append(1.0 - result.score)

        # Return in ChromaDB format (nested lists)
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def count(self) -> int:
        """
        Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

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
            if self.collection_name == collection_name:
                self.collection_name = None
        except Exception:
            # Collection does not exist or error occurred, which is fine
            pass

    def close(self) -> None:
        """Close the database connection."""
        # Qdrant client handles cleanup automatically
        pass
