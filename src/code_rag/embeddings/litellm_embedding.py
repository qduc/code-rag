"""LiteLLM implementation of the embedding interface."""

from typing import List, Optional

from litellm import embedding

from .embedding_interface import EmbeddingInterface


class LiteLLMEmbedding(EmbeddingInterface):
    """LiteLLM implementation for generating embeddings via various providers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        idle_timeout: int = 1800,
    ):
        """
        Initialize the LiteLLM embedding model.

        Args:
            model_name: Name of the model to use (e.g. text-embedding-3-small, vertex_ai/text-embedding-004)
            api_key: API key. If not provided, litellm will look for relevant env vars (OPENAI_API_KEY, etc.)
            idle_timeout: Not used for API-based models (kept for interface consistency)
        """
        self.model_name = model_name
        self.api_key = api_key
        # API-based models don't need to be unloaded, but we store the timeout for consistency
        self._idle_timeout = idle_timeout

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        # Replace newlines to avoid negative performance impact for some models
        text = text.replace("\n", " ")
        response = embedding(model=self.model_name, input=[text], api_key=self.api_key)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            A list of embedding vectors
        """
        # Replace newlines
        texts = [text.replace("\n", " ") for text in texts]
        response = embedding(model=self.model_name, input=texts, api_key=self.api_key)
        # Ensure the order is preserved
        return [data.embedding for data in response.data]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The dimension of embeddings produced by this model
        """
        # Known dimensions for common models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "vertex_ai/text-embedding-004": 768,
            "vertex_ai/text-embedding-gecko": 768,
        }

        if self.model_name in dimensions:
            return dimensions[self.model_name]

        # Fallback: make a dummy call to get dimension
        try:
            return len(self.embed("test"))
        except Exception:
            # Default to 1536 as a safe fallback
            return 1536
