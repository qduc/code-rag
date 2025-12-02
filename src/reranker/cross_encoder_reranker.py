"""Cross-encoder implementation for semantic reranking."""

from typing import List, Tuple, Optional
import threading
from sentence_transformers import CrossEncoder
from .reranker_interface import RerankerInterface


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder implementation using sentence-transformers."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", lazy_load: bool = True):
        """
        Initialize the cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use
            lazy_load: If True, defer model loading until first use (default: True)
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._loading_lock = threading.Lock()
        self._loading_thread: Optional[threading.Thread] = None

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the model (called synchronously or in background)."""
        with self._loading_lock:
            if self.model is None:
                self.model = CrossEncoder(self.model_name)

    def start_background_loading(self):
        """Start loading the model in a background thread."""
        if self._loading_thread is None:
            self._loading_thread = threading.Thread(target=self._load_model, daemon=True)
            self._loading_thread.start()

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use (blocks if still loading)."""
        if self.model is None:
            # Wait for background thread if it exists
            if self._loading_thread is not None:
                self._loading_thread.join()
            # If still not loaded (no background thread was started), load now
            if self.model is None:
                self._load_model()

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
