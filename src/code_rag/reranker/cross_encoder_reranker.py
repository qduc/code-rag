"""Cross-encoder implementation for semantic reranking."""

import gc
import threading
import time
from typing import List, Optional, Tuple

from sentence_transformers import CrossEncoder

from .reranker_interface import RerankerInterface


class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder implementation using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        lazy_load: bool = True,
        idle_timeout: int = 1800,
    ):
        """
        Initialize the cross-encoder model.

        Args:
            model_name: Name of the cross-encoder model to use
            lazy_load: If True, defer model loading until first use (default: True)
            idle_timeout: Seconds of inactivity before auto-unloading model (default: 1800 = 30 min)
        """
        self.model_name = model_name
        self.model: Optional[CrossEncoder] = None
        self._loading_lock = threading.Lock()
        self._loading_thread: Optional[threading.Thread] = None

        # Idle timeout management
        self._idle_timeout = idle_timeout
        self._last_used = time.time()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop_event = threading.Event()

        if not lazy_load:
            self._load_model()

        # Start cleanup thread if idle timeout is enabled
        if self._idle_timeout > 0:
            self._start_cleanup_thread()

    def _load_model(self):
        """Load the model (called synchronously or in background)."""
        with self._loading_lock:
            if self.model is None:
                self.model = CrossEncoder(self.model_name)

    def start_background_loading(self):
        """Start loading the model in a background thread."""
        if self._loading_thread is None:
            self._loading_thread = threading.Thread(
                target=self._load_model, daemon=True
            )
            self._loading_thread.start()

    def _start_cleanup_thread(self):
        """Start the background cleanup thread for idle timeout."""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._cleanup_stop_event.clear()
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name=f"reranker-cleanup-{self.model_name}",
            )
            self._cleanup_thread.start()

    def _cleanup_loop(self):
        """Background loop to check for idle timeout and unload model."""
        while not self._cleanup_stop_event.is_set():
            # Check every minute
            self._cleanup_stop_event.wait(60)

            if self._cleanup_stop_event.is_set():
                break

            # Check if model should be unloaded
            with self._loading_lock:
                if self.model is not None:
                    idle_time = time.time() - self._last_used
                    if idle_time >= self._idle_timeout:
                        # Unload the model
                        del self.model
                        self.model = None
                        # Force garbage collection to release GPU memory if applicable
                        gc.collect()
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except ImportError:
                            pass

    def _ensure_model_loaded(self):
        """Ensure the model is loaded before use (blocks if still loading)."""
        if self.model is None:
            # Wait for background thread if it exists
            if self._loading_thread is not None:
                self._loading_thread.join()
            # If still not loaded (no background thread was started), load now
            if self.model is None:
                self._load_model()

        # Update last used timestamp
        self._last_used = time.time()

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
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
        return scored_indices[: min(top_k, len(scored_indices))]

    def unload_model(self):
        """Explicitly unload the model from memory."""
        with self._loading_lock:
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass

    def stop_cleanup_thread(self):
        """Stop the background cleanup thread."""
        if self._cleanup_thread is not None:
            self._cleanup_stop_event.set()
            self._cleanup_thread.join(timeout=2)

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.stop_cleanup_thread()
        self.unload_model()
