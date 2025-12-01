"""SentenceTransformers implementation of the embedding interface."""

from typing import List

from sentence_transformers import SentenceTransformer

from .embedding_interface import EmbeddingInterface


class SentenceTransformerEmbedding(EmbeddingInterface):
    """SentenceTransformers implementation for generating embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the SentenceTransformers model.

        Args:
            model_name: Name of the model to use (default: all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            A list of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The dimension of embeddings produced by this model
        """
        return self.model.get_sentence_embedding_dimension()
