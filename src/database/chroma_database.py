"""ChromaDB implementation of the database interface."""

from typing import Any, Dict, List, Optional

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
        try:
            self.collection = self.client.get_collection(name=collection_name)

            # Check dimension compatibility
            # 1. Check explicit metadata
            existing_dim = self.collection.metadata.get("dimension")
            stored_model = self.collection.metadata.get("model_name")

            # 2. If no metadata, check existing data
            if existing_dim is None and self.collection.count() > 0:
                peek = self.collection.peek(limit=1)
                if peek["embeddings"]:
                    existing_dim = len(peek["embeddings"][0])

            if existing_dim is not None and int(existing_dim) != vector_size:
                if stored_model:
                    print(
                        f"Dimension mismatch: Collection '{collection_name}' was created "
                        f"with model '{stored_model}' (dimension {existing_dim})."
                    )
                    print("Loading collection with the original model...")
                    return stored_model
                else:
                    # No model name stored, we can't recover gracefully
                    print(
                        f"Dimension mismatch: Collection '{collection_name}' has dimension "
                        f"{existing_dim}, requested {vector_size}."
                    )
                    print(
                        "No model name stored in collection. Use --reindex to recreate with the new model."
                    )
                    raise ValueError(
                        "Dimension mismatch and no model name stored. "
                        "Use --reindex to recreate the collection with the new model."
                    )

            return None

        except ValueError:
            # Re-raise ValueError for dimension mismatch without model name
            raise
        except Exception:
            # Collection does not exist or other error occurred
            self.collection = None

        if self.collection is None:
            metadata = {"hnsw:space": "cosine", "dimension": vector_size}
            if model_name:
                metadata["model_name"] = model_name
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata,
            )

        return None

    def get_model_name(self) -> Optional[str]:
        """
        Get the model name stored in the collection metadata.

        Returns:
            The model name if stored, None otherwise
        """
        if self.collection is None:
            return None
        return self.collection.metadata.get("model_name")

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

    def query(self, embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
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

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by their IDs.

        Args:
            ids: List of document IDs to delete
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        if not ids:
            return

        try:
            self.collection.delete(ids=ids)
        except Exception as e:
            print(f"Error deleting documents: {e}")
            raise

    def get_all_ids(self) -> List[str]:
        """
        Get all document IDs in the collection.

        Returns:
            List of all document IDs
        """
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        try:
            # Get all documents with just IDs
            results = self.collection.get(include=[])
            return results.get("ids", [])
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
        if self.collection is None:
            raise RuntimeError("Collection not initialized. Call initialize() first.")

        try:
            # Try using where clause to filter by file_path
            results = self.collection.get(where={"file_path": file_path}, include=[])
            return results.get("ids", [])
        except Exception:
            # ChromaDB might not support this where clause, fallback to manual filter
            try:
                all_results = self.collection.get(include=["metadatas"])
                ids = []
                for i, metadata in enumerate(all_results.get("metadatas", [])):
                    if metadata.get("file_path") == file_path:
                        ids.append(all_results["ids"][i])
                return ids
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
            if self.collection and self.collection.name == collection_name:
                self.collection = None
        except (ValueError, NotFoundError):
            # Collection does not exist, which is fine
            pass

    def close(self) -> None:
        """Close the database connection."""
        # ChromaDB PersistentClient handles persistence automatically
        pass
