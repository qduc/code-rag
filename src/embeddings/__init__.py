"""Embeddings module for generating vector embeddings."""

from .embedding_interface import EmbeddingInterface
from .sentence_transformer_embedding import SentenceTransformerEmbedding

__all__ = ["EmbeddingInterface", "SentenceTransformerEmbedding"]