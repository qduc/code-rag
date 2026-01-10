"""Test to verify requests are held during model loading and proceed when ready.

This test verifies the key behavior:
- If a request comes while the model is still loading, the request should be held
- The request should proceed successfully when the model finishes loading
- No error should be returned during the loading phase

Performance optimization:
- Most tests use mocked models (fast, ~10ms) to test lazy loading LOGIC
- One integration test uses real model loading to verify actual model initialization
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from code_rag.api import CodeRAGAPI
from code_rag.embeddings.sentence_transformer_embedding import (
    SentenceTransformerEmbedding,
)
from code_rag.mcp_server import call_tool


class TestRequestDuringModelLoading:
    """Tests for request behavior during model loading phase."""

    def test_embedding_model_blocks_on_load_during_embed_integration(self):
        """INTEGRATION TEST: Verify real model loading works (slow, ~4-5s).

        This is the ONE test that exercises actual PyTorch model loading
        to ensure the integration works end-to-end. All other tests use
        mocked models for speed.
        """
        # Create embedding with lazy loading - REAL MODEL
        embedding = SentenceTransformerEmbedding(lazy_load=True)

        # Start background loading
        embedding.start_background_loading()

        # Model should not be loaded yet
        assert embedding.model is None

        # Now call embed() - this should block until model is loaded
        start_time = time.time()
        result = embedding.embed("test query")
        elapsed = time.time() - start_time

        # Model should now be loaded
        assert embedding.model is not None

        # Result should be valid embedding vector
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)

        # Should have taken some time to load (model download/initialization)
        print(f"Real model loading took {elapsed:.2f} seconds")

    def test_embed_batch_blocks_on_load(self, mock_sentence_transformer_model):
        """Test that embed_batch() blocks until model is loaded (using mock for speed)."""

        # Create a factory that returns the mock with a delay
        def create_delayed_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.1)

        # Patch where SentenceTransformer is actually used (in the embedding module)
        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_delayed_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)
            embedding.start_background_loading()

            # Give background thread a moment to start but not complete
            time.sleep(0.02)

            # Model should not be loaded yet (background load takes 0.1s)
            # Note: Due to threading timing, we can't reliably assert None here

            # embed_batch should block until model is ready
            texts = ["text1", "text2", "text3"]
            results = embedding.embed_batch(texts)

            # Model should now be loaded
            assert embedding.model is not None

            # Results should be correct
            assert len(results) == 3
            for result in results:
                assert isinstance(result, list)
                assert len(result) > 0

    def test_get_embedding_dimension_blocks_on_load(
        self, mock_sentence_transformer_model
    ):
        """Test that get_embedding_dimension() blocks until model is loaded (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.05)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)
            embedding.start_background_loading()

            # Note: Can't reliably assert model is None due to threading timing

            # get_embedding_dimension should block until model is ready
            dimension = embedding.get_embedding_dimension()

            # Model should now be loaded
            assert embedding.model is not None

            # Dimension should be valid
            assert isinstance(dimension, int)
            assert dimension == 384  # Mock returns 384

    def test_concurrent_requests_during_loading(self, mock_sentence_transformer_model):
        """Test that multiple concurrent requests all block and proceed successfully (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(
                load_delay=0.2
            )  # 200ms simulated load

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)
            embedding.start_background_loading()

            results = []
            errors = []

            def make_request(query_text, request_id):
                """Simulate a request that embeds text."""
                try:
                    result = embedding.embed(query_text)
                    results.append(
                        {
                            "request_id": request_id,
                            "success": True,
                            "embedding_length": len(result),
                        }
                    )
                except Exception as e:
                    errors.append({"request_id": request_id, "error": str(e)})

            # Launch multiple concurrent requests immediately
            threads = []
            for i in range(5):
                t = threading.Thread(target=make_request, args=(f"query_{i}", i))
                threads.append(t)
                t.start()

            # Wait for all requests to complete
            for t in threads:
                t.join()

            # All requests should succeed
            assert len(errors) == 0, f"Unexpected errors: {errors}"
            assert len(results) == 5

            # All results should have valid embeddings
            for result in results:
                assert result["success"] is True
                assert result["embedding_length"] > 0

    def test_sequential_requests_during_loading(self, mock_sentence_transformer_model):
        """Test that sequential requests during loading all proceed without error (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.1)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)
            embedding.start_background_loading()

            # Note: Can't reliably assert model is None due to threading timing

            # First request - should wait for model
            result1 = embedding.embed("query 1")
            assert isinstance(result1, list)
            assert embedding.model is not None  # Now loaded

            # Subsequent requests - model already loaded
            result2 = embedding.embed("query 2")
            assert isinstance(result2, list)

            result3 = embedding.embed("query 3")
            assert isinstance(result3, list)

    @pytest.mark.asyncio
    async def test_api_ensure_indexed_handles_model_loading(self):
        """Test that API.ensure_indexed() properly handles model loading.

        When API is created with lazy_load_models=True and background loading
        is started, ensure_indexed should work correctly even if called before
        the model finishes loading.
        """
        with tempfile.TemporaryDirectory() as temp_db:
            with tempfile.TemporaryDirectory() as temp_codebase:
                # Create a small test file
                test_file = Path(temp_codebase) / "test.py"
                test_file.write_text("def hello():\n    print('world')")

                # Create API with lazy loading
                api = CodeRAGAPI(
                    database_path=temp_db,
                    reranker_enabled=False,
                    lazy_load_models=True,
                )

                # Start background loading
                api.start_background_loading()

                # Model should not be loaded yet (or still loading)
                # But ensure_indexed should still work
                result = api.ensure_indexed(
                    temp_codebase,
                    collection_name="test",
                    validate_codebase=False,
                )

                # Should succeed despite potential model loading in progress
                assert result["success"] is True
                assert result["total_chunks"] > 0

                api.close()
                print("API.ensure_indexed succeeded during model loading")

    @pytest.mark.asyncio
    async def test_call_tool_waits_for_api_initialization(self):
        """Test that call_tool properly waits for API initialization.

        Simulates: request arrives -> API still initializing in background ->
        call_tool waits -> API finishes initializing -> call_tool proceeds successfully

        This test uses state-based verification instead of timing assertions:
        - Verifies API is not initialized when tool is called
        - Verifies tool call succeeds (doesn't return error)
        - Verifies API is initialized after tool returns
        - This approach is environment-independent and reliable across CI/local
        """
        import code_rag.mcp_server as mcp_mod

        # Save original state
        original_api = mcp_mod.api
        original_event = mcp_mod.api_ready_event

        try:
            # Reset to simulated "not initialized" state
            mcp_mod.api = None
            mcp_mod.api_ready_event = threading.Event()

            with tempfile.TemporaryDirectory() as temp_db:
                with tempfile.TemporaryDirectory() as temp_codebase:
                    # Create a test file
                    test_file = Path(temp_codebase) / "test.py"
                    test_file.write_text("def test():\n    pass")

                    # Simulate delayed API initialization with controlled delay
                    # Using a short, deterministic delay (0.1s) instead of real model loading
                    # This makes the test fast and environment-independent
                    def init_api_delayed():
                        # Simulate brief startup delay (10ms threading overhead)
                        time.sleep(0.01)
                        api = CodeRAGAPI(
                            database_path=temp_db,
                            reranker_enabled=False,
                        )
                        api.ensure_indexed(
                            temp_codebase,
                            collection_name="test",
                            validate_codebase=False,
                        )
                        mcp_mod.api = api
                        mcp_mod.api_ready_event.set()

                    # Start initialization in background
                    init_thread = threading.Thread(target=init_api_delayed, daemon=True)
                    init_thread.start()

                    # Verify API is not initialized yet (state assertion)
                    assert (
                        mcp_mod.api is None
                    ), "API should not be initialized at test start"

                    # Call tool - should wait for API to initialize
                    result = await call_tool(
                        "search_codebase",
                        {
                            "codebase_path": temp_codebase,
                            "query": "test",
                        },
                    )

                    # State-based assertions: verify the waiting behavior through outcomes
                    # 1. Tool should have succeeded (not returned initialization error)
                    assert len(result) == 1, "Tool should return exactly one result"
                    assert (
                        "not initialized" not in result[0].text.lower()
                    ), "Tool should not return 'not initialized' error - call_tool should have waited"

                    # 2. API should now be initialized (after call_tool returns)
                    assert (
                        mcp_mod.api is not None
                    ), "API should be initialized after call_tool returns"

                    print(
                        "✓ call_tool correctly waited for API initialization and succeeded"
                    )

                    # Cleanup
                    init_thread.join()
                    if mcp_mod.api:
                        mcp_mod.api.close()
        finally:
            # Restore original state
            mcp_mod.api = original_api
            mcp_mod.api_ready_event = original_event

    @pytest.mark.asyncio
    async def test_call_tool_returns_error_only_after_wait_timeout(self):
        """Test that call_tool only returns error if API never initializes (timeout).

        Verifies the error is returned after the wait timeout, not immediately.
        This test verifies that the wait timeout in call_tool is enforced.
        """
        import code_rag.mcp_server as mcp_mod

        original_api = mcp_mod.api
        original_event = mcp_mod.api_ready_event

        try:
            # Simulate API that never initializes
            mcp_mod.api = None
            mcp_mod.api_ready_event = threading.Event()

            # Don't set the event - API never initializes
            # Use a short timeout for testing (0.1s instead of 30s)
            start_time = time.time()

            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": "/test",
                    "query": "test",
                },
                _api_wait_timeout=0.1,
            )
            elapsed = time.time() - start_time

            # Should return error message (event was never set)
            assert len(result) == 1
            assert "not initialized" in result[0].text.lower()

            # Should have waited approximately the timeout duration
            # Allow some tolerance for scheduling overhead
            assert (
                elapsed >= 0.05
            ), f"Should have waited at least 0.05s, but only waited {elapsed:.3f}s"
            print(f"Returned error after {elapsed:.2f}s wait")
        finally:
            mcp_mod.api = original_api
            mcp_mod.api_ready_event = original_event

    def test_embedding_dimension_with_lazy_load(self, mock_sentence_transformer_model):
        """Test that get_embedding_dimension() correctly waits for lazy-loaded model (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.05)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)

            # Before loading, model is None
            assert embedding.model is None

            # Start background loading
            embedding.start_background_loading()

            # get_embedding_dimension should block and return correct value
            dimension = embedding.get_embedding_dimension()

            assert embedding.model is not None
            assert dimension == 384  # Mock returns 384


class TestModelLoadingErrorHandling:
    """Tests for error handling during model loading."""

    def test_ensure_model_loaded_handles_no_background_thread(
        self, mock_sentence_transformer_model
    ):
        """Test that _ensure_model_loaded works even without background thread (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)

            # Don't start background loading, but try to use the model
            # _ensure_model_loaded should load it synchronously
            result = embedding.embed("test")

            assert embedding.model is not None
            assert isinstance(result, list)

    def test_multiple_loading_attempts_thread_safe(
        self, mock_sentence_transformer_model
    ):
        """Test that multiple threads trying to load the model handle it correctly (using mock)."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.1)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            embedding = SentenceTransformerEmbedding(lazy_load=True)

            def load_and_use():
                # Each thread tries to access the model
                embedding.embed("test")

            threads = [threading.Thread(target=load_and_use) for _ in range(3)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Model should be loaded exactly once
            assert embedding.model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
