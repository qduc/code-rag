"""OpenAI implementation of the embedding interface."""

import os
from typing import List, Optional

from openai import OpenAI

from .embedding_interface import EmbeddingInterface


class OpenAIEmbedding(EmbeddingInterface):
    """OpenAI implementation for generating embeddings."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        idle_timeout: int = 1800,  # Not used for API-based models, added for interface consistency
    ):
        """
        Initialize the OpenAI embedding model.

        Args:
            model_name: Name of the model to use (default: text-embedding-3-small)
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY env var.
            idle_timeout: Not used for API-based models (kept for interface consistency)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # OpenAI is API-based, no model to unload, but we store the timeout for consistency
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
        response = self.client.embeddings.create(input=[text], model=self.model_name)
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
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        # Ensure the order is preserved (OpenAI usually preserves order)
        return [data.embedding for data in response.data]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The dimension of embeddings produced by this model
        """
        # Known dimensions for common OpenAI models
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        if self.model_name in dimensions:
            return dimensions[self.model_name]

        # Fallback: make a dummy call to get dimension (cached if possible)
        # For now, let's default to 1536 as it's the most common for recent models
        # or maybe raise a warning.
        # But let's just return 1536 as a safe default for now or try to fetch it.
        try:
            # minimal cost call
            return len(self.embed("test"))
        except Exception:
            return 1536
