"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingInterface(ABC):
    """Abstract interface for embedding models."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        pass

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string with the appropriate instruction prefix.

        This method allows models to apply special query-specific prefixes.
        Default implementation delegates to embed(), but subclasses can override.

        Args:
            query: The query text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        return self.embed(query)

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            A list of embedding vectors
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The dimension of embeddings produced by this model
        """
        pass

    def clear_cache(self):
        """
        Clear memory cache (e.g. CUDA memory) without unloading the model.
        Useful after memory-intensive operations like indexing.
        """
        return
