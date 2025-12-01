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
