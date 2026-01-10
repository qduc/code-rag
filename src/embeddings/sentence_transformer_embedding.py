"""SentenceTransformers implementation of the embedding interface."""

import gc
import threading
import time
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .embedding_interface import EmbeddingInterface


class SentenceTransformerEmbedding(EmbeddingInterface):
    """SentenceTransformers implementation for generating embeddings."""

    # Models that require special instruction prefixes for queries
    QUERY_INSTRUCTION_PREFIX = {
        "nomic-ai/CodeRankEmbed": "Represent this query for searching relevant code: ",
    }

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        lazy_load: bool = False,
        idle_timeout: int = 1800,
    ):
        """
        Initialize the SentenceTransformers model.

        Args:
            model_name: Name of the model to use (default: all-MiniLM-L6-v2)
            lazy_load: If True, defer model loading until first use
            idle_timeout: Seconds of inactivity before auto-unloading model (default: 1800 = 30 min)
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._loading_lock = threading.Lock()
        self._loading_thread: Optional[threading.Thread] = None

        # Idle timeout management
        self._idle_timeout = idle_timeout
        self._last_used = time.time()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop_event = threading.Event()

        # Determine if this model requires query instruction prefix
        self.query_prefix = self.QUERY_INSTRUCTION_PREFIX.get(model_name, "")

        if not lazy_load:
            self._load_model()

        # Start cleanup thread if idle timeout is enabled
        if self._idle_timeout > 0:
            self._start_cleanup_thread()

    def _load_model(self):
        """Load the model (called synchronously or in background)."""
        with self._loading_lock:
            if self.model is None:
                self.model = SentenceTransformer(
                    self.model_name, trust_remote_code=True
                )

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
                name=f"model-cleanup-{self.model_name}",
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

    def embed(self, text: str) -> List[float]:
        """
        Embed a single text string into a vector.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        self._ensure_model_loaded()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a query string with the appropriate instruction prefix.

        For models like CodeRankEmbed that require special query instructions,
        this method prepends the required prefix.

        Args:
            query: The query text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        if self.query_prefix:
            query = self.query_prefix + query
        return self.embed(query)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple text strings into vectors.

        Args:
            texts: List of texts to embed

        Returns:
            A list of embedding vectors
        """
        self._ensure_model_loaded()
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The dimension of embeddings produced by this model
        """
        self._ensure_model_loaded()
        return self.model.get_sentence_embedding_dimension()

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

    def clear_cache(self):
        """Clear memory cache (CUDA memory) without unloading the model."""
        with self._loading_lock:
            # Force garbage collection
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
