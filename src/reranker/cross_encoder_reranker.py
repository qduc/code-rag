"""Cross-encoder implementation for semantic reranking."""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from .reranker_interface import RerankerInterface


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder implementation using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None  # Lazy loading

    def _ensure_model_loaded(self):
        """Lazy load the model on first use."""
        if self.model is None:
            self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using cross-encoder scoring.

        Args:
            query: The search query string
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        self._ensure_model_loaded()

        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs
        scores = self.model.predict(pairs, convert_to_numpy=True)

        # Create (index, score) tuples and sort by score (descending)
        scored_indices = [(i, float(scores[i])) for i in range(len(scores))]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        # Return top_k results
        return scored_indices[:min(top_k, len(scored_indices))]
