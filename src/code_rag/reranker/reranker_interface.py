"""Abstract base class for semantic rerankers."""

from abc import ABC, abstractmethod
from typing import List, Tuple


class RerankerInterface(ABC):
    """Abstract interface for semantic rerankers."""

    @abstractmethod
    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query string
            documents: List of document texts to rerank
            top_k: Number of top results to return after reranking

        Returns:
            List of tuples (original_index, relevance_score) sorted by relevance (highest first)
            relevance_score is a float where higher = more relevant
        """
        pass
