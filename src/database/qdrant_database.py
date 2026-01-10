"""Qdrant implementation of the database interface."""

import json
import os
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .database_interface import DatabaseInterface


class QdrantDatabase(DatabaseInterface):
    """Qdrant implementation for vector storage."""

    def __init__(
        self,
        persist_directory: str = ".code-rag",
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
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
        self._model_name = None

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
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._model_name = model_name

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

            # Get stored model name from collection metadata
            stored_model = self._get_stored_model_name(collection_name)

            if current_size is not None and current_size != vector_size:
                if stored_model:
                    print(
                        f"Dimension mismatch: Collection '{collection_name}' was created "
                        f"with model '{stored_model}' (dimension {current_size})."
                    )
                    print("Loading collection with the original model...")
                    self._model_name = stored_model
                    self.vector_size = current_size
                    return stored_model
                else:
                    # No model name stored, we can't recover gracefully
                    print(
                        f"Dimension mismatch: Collection '{collection_name}' has dimension "
                        f"{current_size}, requested {vector_size}."
                    )
                    print(
                        "No model name stored in collection. Use --reindex to recreate with the new model."
                    )
                    raise ValueError(
                        "Dimension mismatch and no model name stored. "
                        "Use --reindex to recreate the collection with the new model."
                    )
            else:
                # Dimensions match, use stored model name if available
                if stored_model:
                    self._model_name = stored_model

            return None

        # Create collection with cosine distance
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        # Store model name in a metadata file (Qdrant doesn't have collection-level metadata)
        if model_name:
            self._store_model_name(collection_name, model_name)

        return None

    def _get_metadata_path(self, collection_name: str) -> str:
        """Get the path to the metadata file for a collection."""
        return os.path.join(self.persist_directory, f"{collection_name}_metadata.json")

    def _store_model_name(self, collection_name: str, model_name: str) -> None:
        """Store model name in a metadata file."""
        metadata_path = self._get_metadata_path(collection_name)
        metadata = {"model_name": model_name}
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

    def _get_stored_model_name(self, collection_name: str) -> Optional[str]:
        """Get the stored model name from metadata file."""
        metadata_path = self._get_metadata_path(collection_name)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    return metadata.get("model_name")
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def get_model_name(self) -> Optional[str]:
        """
        Get the model name stored in the collection metadata.

        Returns:
            The model name if stored, None otherwise
        """
        if self.collection_name:
            return self._get_stored_model_name(self.collection_name)
        return self._model_name

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
        for i, (doc_id, embedding, document) in enumerate(
            zip(ids, embeddings, documents)
        ):
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

    def query(self, embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
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

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        if not ids:
            return

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
                wait=True,
            )
        except Exception as e:
            print(f"Error deleting points: {e}")
            raise

    def get_all_ids(self) -> List[str]:
        """
        Get all document IDs in the collection.

        Returns:
            List of all document IDs
        """
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        try:
            # Use scroll to get all points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on expected collection size
                with_payload=False,
                with_vectors=False,
            )
            return [str(point.id) for point in points]
        except Exception as e:
            print(f"Error getting all IDs: {e}")
            return []

    def get_ids_by_file(self, file_path: str) -> List[str]:
        """
        Get all document IDs for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            List of document IDs for that file
        """
        if self.collection_name is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        try:
            # Use scroll with filter
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path", match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )
            return [str(point.id) for point in points]
        except Exception as e:
            print(f"Error getting IDs by file: {e}")
            return []

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
            # Also delete the metadata file
            metadata_path = self._get_metadata_path(collection_name)
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        except Exception:
            # Collection does not exist or error occurred, which is fine
            pass

    def close(self) -> None:
        """Close the database connection."""
        # Qdrant client handles cleanup automatically
        pass
