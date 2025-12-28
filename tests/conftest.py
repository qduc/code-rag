"""Pytest configuration and shared fixtures for the tests package."""

from pathlib import Path
import sys
import tempfile

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))  # Ensure src/ is importable when running tests


def pytest_configure(config):
    """Register custom markers used throughout the tests."""
    config.addinivalue_line("markers", "asyncio: mark test as async")

    # Disable shared server mode during tests to avoid connection issues
    import os
    os.environ["CODE_RAG_SHARED_SERVER"] = "false"


@pytest.fixture(scope="session")
def event_loop():
    """Provide a dedicated event loop for the entire pytest session."""
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def mock_sentence_transformer_model():
    """Provide a lightweight mock of SentenceTransformer model for fast testing.

    This mock simulates the essential behavior of a sentence-transformers model
    without the 4-5 second load time. It's designed for testing lazy loading logic,
    concurrency, and error handling without actually loading PyTorch weights.

    Returns a factory function that creates mock models with controllable behavior.
    """
    import numpy as np
    from unittest.mock import MagicMock

    def create_mock_model(embedding_dim=384, load_delay=0.0):
        """Create a mock model that behaves like SentenceTransformer.

        Args:
            embedding_dim: Dimension of embedding vectors to return
            load_delay: Simulated load time in seconds (for testing wait behavior)
        """
        import time

        mock_model = MagicMock()

        # Simulate model attributes
        mock_model.get_sentence_embedding_dimension.return_value = embedding_dim

        # Mock encode method to return deterministic embeddings
        def mock_encode(texts, **kwargs):
            if load_delay > 0:
                time.sleep(load_delay)

            if isinstance(texts, str):
                texts = [texts]

            # Return deterministic embeddings based on text hash
            embeddings = []
            for text in texts:
                # Create a deterministic but varied embedding
                seed = hash(text) % (2**31)
                np.random.seed(seed)
                embedding = np.random.randn(embedding_dim).astype(np.float32)
                # Normalize to unit length (like real sentence-transformers)
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            # Return single array for single input, 2D numpy array for batch
            if len(embeddings) > 1:
                return np.array(embeddings)  # 2D array for batch
            else:
                return embeddings[0]  # 1D array for single

        mock_model.encode = mock_encode

        return mock_model

    return create_mock_model


@pytest.fixture(scope="session")
def shared_temp_db():
    """Provide a shared temporary database directory for the entire test session."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup happens after all tests complete
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
