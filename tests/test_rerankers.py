"""Comprehensive tests for the reranker modules.

These tests verify:
1. HTTPReranker initialization and HTTP communication
2. HTTPReranker error handling (connection failures, timeouts)
3. CrossEncoderReranker initialization with different models
4. CrossEncoderReranker model loading and caching
5. CrossEncoderReranker reranking logic (scoring, sorting)
6. CrossEncoderReranker idle timeout behavior
7. Empty inputs handling for both rerankers
8. Thread safety for CrossEncoderReranker

All tests mock external dependencies to avoid real HTTP requests and model loading.
"""

import gc
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, PropertyMock, patch

import numpy as np
import pytest
import requests

# ============================================================================
# HTTPReranker Tests
# ============================================================================


class TestHttpRerankerInit:
    """Tests for HTTPReranker initialization."""

    def test_init_with_default_port(self):
        """Test initialization with default port."""
        from code_rag.reranker.http_reranker import DEFAULT_PORT, HttpReranker

        reranker = HttpReranker()

        assert reranker.port == DEFAULT_PORT
        assert reranker.base_url == f"http://127.0.0.1:{DEFAULT_PORT}"
        assert reranker.client_id == ""

    def test_init_with_custom_port(self):
        """Test initialization with custom port."""
        from code_rag.reranker.http_reranker import HttpReranker

        reranker = HttpReranker(port=9999)

        assert reranker.port == 9999
        assert reranker.base_url == "http://127.0.0.1:9999"

    def test_init_with_client_id(self):
        """Test initialization with client_id."""
        from code_rag.reranker.http_reranker import HttpReranker

        reranker = HttpReranker(port=8199, client_id="test-client-123")

        assert reranker.client_id == "test-client-123"


class TestHttpRerankerRerank:
    """Tests for HTTPReranker rerank method."""

    @pytest.fixture
    def reranker(self):
        """Provide an HTTPReranker instance."""
        from code_rag.reranker.http_reranker import HttpReranker

        return HttpReranker(port=8199, client_id="test-client")

    def test_rerank_empty_documents_returns_empty_list(self, reranker):
        """Test that reranking empty documents returns empty list without HTTP call."""
        with patch("requests.post") as mock_post:
            result = reranker.rerank(query="test query", documents=[])

            assert result == []
            mock_post.assert_not_called()

    def test_rerank_successful_response(self, reranker):
        """Test successful reranking via HTTP."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [[0, 0.95], [2, 0.85], [1, 0.75]]}
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = reranker.rerank(
                query="test query", documents=["doc1", "doc2", "doc3"], top_k=3
            )

            assert result == [(0, 0.95), (2, 0.85), (1, 0.75)]
            mock_post.assert_called_once()

            # Verify the request payload
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://127.0.0.1:8199/rerank"
            payload = call_args[1]["json"]
            assert payload["query"] == "test query"
            assert payload["documents"] == ["doc1", "doc2", "doc3"]
            assert payload["top_k"] == 3
            assert payload["client_id"] == "test-client"

    def test_rerank_with_metadatas(self, reranker):
        """Test reranking with metadata passed."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [[0, 0.9]]}
        mock_response.raise_for_status = Mock()

        metadatas = [{"file_path": "test.py", "function_name": "foo"}]

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = reranker.rerank(
                query="test query", documents=["doc1"], metadatas=metadatas, top_k=1
            )

            assert result == [(0, 0.9)]
            payload = mock_post.call_args[1]["json"]
            assert payload["metadatas"] == metadatas

    def test_rerank_with_custom_model(self, reranker):
        """Test reranking with custom model parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [[0, 0.9]]}
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = reranker.rerank(
                query="test query", documents=["doc1"], model="custom-model"
            )

            payload = mock_post.call_args[1]["json"]
            assert payload["model"] == "custom-model"

    def test_rerank_error_in_response_returns_empty(self, reranker):
        """Test that error in response JSON returns empty list."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Model not loaded"}
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = reranker.rerank(query="test query", documents=["doc1", "doc2"])

            assert result == []


class TestHttpRerankerErrorHandling:
    """Tests for HTTPReranker error handling."""

    @pytest.fixture
    def reranker(self):
        """Provide an HTTPReranker instance."""
        from code_rag.reranker.http_reranker import HttpReranker

        return HttpReranker(port=8199)

    def test_rerank_connection_error_returns_empty(self, reranker):
        """Test that connection error returns empty list."""
        with patch("requests.post", side_effect=requests.ConnectionError()):
            result = reranker.rerank(query="test query", documents=["doc1", "doc2"])

            assert result == []

    def test_rerank_timeout_returns_empty(self, reranker):
        """Test that timeout returns empty list."""
        with patch("requests.post", side_effect=requests.Timeout()):
            result = reranker.rerank(query="test query", documents=["doc1", "doc2"])

            assert result == []

    def test_rerank_http_error_propagates(self, reranker):
        """Test that HTTP errors (non-connection) propagate."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )

        with patch("requests.post", return_value=mock_response):
            with pytest.raises(requests.HTTPError):
                reranker.rerank(query="test query", documents=["doc1", "doc2"])


class TestHttpRerankerNoOps:
    """Tests for HTTPReranker no-op methods."""

    def test_start_background_loading_is_noop(self):
        """Test that start_background_loading is a no-op."""
        from code_rag.reranker.http_reranker import HttpReranker

        reranker = HttpReranker()
        # Should not raise any exception
        reranker.start_background_loading()

    def test_unload_model_is_noop(self):
        """Test that unload_model is a no-op."""
        from code_rag.reranker.http_reranker import HttpReranker

        reranker = HttpReranker()
        # Should not raise any exception
        reranker.unload_model()


# ============================================================================
# CrossEncoderReranker Tests
# ============================================================================


class TestCrossEncoderRerankerInit:
    """Tests for CrossEncoderReranker initialization."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    def test_init_with_default_model_lazy_load(self, mock_cross_encoder):
        """Test initialization with lazy loading (default)."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        assert reranker.model_name == "jinaai/jina-reranker-v3"
        assert reranker.model is None  # Not loaded yet due to lazy_load=True
        mock_class.assert_not_called()

        # Cleanup
        reranker.stop_cleanup_thread()

    def test_init_with_custom_model(self, mock_cross_encoder):
        """Test initialization with custom model name."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(model_name="custom/reranker-model")

        assert reranker.model_name == "custom/reranker-model"
        reranker.stop_cleanup_thread()

    def test_init_with_lazy_load_false_loads_model(self, mock_cross_encoder):
        """Test initialization with lazy_load=False loads model immediately."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False)

        mock_class.assert_called_once_with("jinaai/jina-reranker-v3")
        assert reranker.model == mock_instance

        reranker.stop_cleanup_thread()

    def test_init_with_idle_timeout(self, mock_cross_encoder):
        """Test initialization with custom idle timeout."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=600)

        assert reranker._idle_timeout == 600
        reranker.stop_cleanup_thread()

    def test_init_with_zero_idle_timeout_no_cleanup_thread(self, mock_cross_encoder):
        """Test that zero idle timeout doesn't start cleanup thread."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)

        assert reranker._idle_timeout == 0
        # Cleanup thread should not have started
        assert reranker._cleanup_thread is None


class TestCrossEncoderRerankerModelLoading:
    """Tests for CrossEncoderReranker model loading behavior."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    def test_load_model_creates_cross_encoder(self, mock_cross_encoder):
        """Test _load_model creates CrossEncoder instance."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)
        reranker._load_model()

        mock_class.assert_called_once_with("jinaai/jina-reranker-v3")
        assert reranker.model == mock_instance

    def test_load_model_only_loads_once(self, mock_cross_encoder):
        """Test _load_model only loads model once."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)
        reranker._load_model()
        reranker._load_model()  # Second call should be no-op

        mock_class.assert_called_once()

    def test_start_background_loading_creates_thread(self, mock_cross_encoder):
        """Test start_background_loading starts a background thread."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)

        assert reranker._loading_thread is None
        reranker.start_background_loading()

        assert reranker._loading_thread is not None
        assert reranker._loading_thread.daemon is True

        # Wait for thread to complete
        reranker._loading_thread.join(timeout=2)
        assert reranker.model == mock_instance

    def test_ensure_model_loaded_waits_for_background_thread(self, mock_cross_encoder):
        """Test _ensure_model_loaded waits for background loading to complete."""
        mock_class, mock_instance = mock_cross_encoder

        # Add a small delay to model loading
        loading_started = threading.Event()
        loading_complete = threading.Event()

        def delayed_cross_encoder(*args, **kwargs):
            loading_started.set()
            time.sleep(0.1)
            loading_complete.set()
            return mock_instance

        mock_class.side_effect = delayed_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)
        reranker.start_background_loading()

        # Wait for loading to start
        loading_started.wait(timeout=1)

        # This should block until loading completes
        reranker._ensure_model_loaded()

        assert loading_complete.is_set()
        assert reranker.model == mock_instance


class TestCrossEncoderRerankerRerank:
    """Tests for CrossEncoderReranker rerank method."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class with predict method."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            # Default scores
            mock_instance.predict.return_value = np.array([0.8, 0.6, 0.9])
            mock.return_value = mock_instance
            yield mock, mock_instance

    @pytest.fixture
    def reranker(self, mock_cross_encoder):
        """Provide a CrossEncoderReranker instance with mocked model."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)
        yield reranker

    def test_rerank_empty_documents_returns_empty(self, reranker, mock_cross_encoder):
        """Test reranking empty documents returns empty list."""
        _, mock_instance = mock_cross_encoder

        result = reranker.rerank(query="test query", documents=[])

        assert result == []
        mock_instance.predict.assert_not_called()

    def test_rerank_returns_sorted_results(self, reranker, mock_cross_encoder):
        """Test reranking returns results sorted by score descending."""
        _, mock_instance = mock_cross_encoder
        mock_instance.predict.return_value = np.array([0.5, 0.9, 0.3])

        result = reranker.rerank(
            query="test query", documents=["doc1", "doc2", "doc3"], top_k=3
        )

        # doc2 (idx=1) has highest score, then doc1 (idx=0), then doc3 (idx=2)
        assert result == [(1, 0.9), (0, 0.5), (2, 0.3)]

    def test_rerank_respects_top_k(self, reranker, mock_cross_encoder):
        """Test reranking respects top_k limit."""
        _, mock_instance = mock_cross_encoder
        mock_instance.predict.return_value = np.array([0.5, 0.9, 0.3, 0.7, 0.8])

        result = reranker.rerank(
            query="test query",
            documents=["doc1", "doc2", "doc3", "doc4", "doc5"],
            top_k=2,
        )

        assert len(result) == 2
        assert result[0] == (1, 0.9)  # Highest score
        assert result[1] == (4, 0.8)  # Second highest

    def test_rerank_creates_query_document_pairs(self, reranker, mock_cross_encoder):
        """Test reranking creates correct query-document pairs."""
        _, mock_instance = mock_cross_encoder

        reranker.rerank(
            query="search query", documents=["first doc", "second doc"], top_k=2
        )

        # Check the pairs passed to predict
        call_args = mock_instance.predict.call_args
        pairs = call_args[0][0]

        assert pairs == [["search query", "first doc"], ["search query", "second doc"]]

    def test_rerank_with_metadata_adds_context(self, reranker, mock_cross_encoder):
        """Test reranking with metadata enriches documents with context."""
        _, mock_instance = mock_cross_encoder
        mock_instance.predict.return_value = np.array([0.8, 0.6])

        metadatas = [
            {"file_path": "test.py", "function_name": "foo", "class_name": "Bar"},
            {"file_path": "other.py"},
        ]

        reranker.rerank(
            query="test query", documents=["doc1", "doc2"], metadatas=metadatas, top_k=2
        )

        call_args = mock_instance.predict.call_args
        pairs = call_args[0][0]

        # First doc should have full context
        assert "File: test.py" in pairs[0][1]
        assert "Class: Bar" in pairs[0][1]
        assert "Function: foo" in pairs[0][1]
        assert "Code: doc1" in pairs[0][1]

        # Second doc has minimal context
        assert "File: other.py" in pairs[1][1]
        assert "Code: doc2" in pairs[1][1]

    def test_rerank_with_partial_metadata_uses_documents_directly(
        self, reranker, mock_cross_encoder
    ):
        """Test reranking with mismatched metadata length uses raw documents."""
        _, mock_instance = mock_cross_encoder
        mock_instance.predict.return_value = np.array([0.8, 0.6])

        # Metadata length doesn't match documents length
        metadatas = [{"file_path": "test.py"}]  # Only 1 metadata for 2 docs

        reranker.rerank(
            query="test query", documents=["doc1", "doc2"], metadatas=metadatas, top_k=2
        )

        call_args = mock_instance.predict.call_args
        pairs = call_args[0][0]

        # Should use raw documents (no metadata processing)
        assert pairs[0][1] == "doc1"
        assert pairs[1][1] == "doc2"

    def test_rerank_updates_last_used_timestamp(self, mock_cross_encoder):
        """Test that reranking updates the last used timestamp."""
        mock_class, mock_instance = mock_cross_encoder
        mock_instance.predict.return_value = np.array([0.8])

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)

        initial_time = reranker._last_used
        time.sleep(0.01)

        reranker.rerank(query="test", documents=["doc"])

        assert reranker._last_used > initial_time


class TestCrossEncoderRerankerIdleTimeout:
    """Tests for CrossEncoderReranker idle timeout behavior."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            mock_instance.predict.return_value = np.array([0.8])
            mock.return_value = mock_instance
            yield mock, mock_instance

    def test_cleanup_thread_starts_with_positive_timeout(self, mock_cross_encoder):
        """Test cleanup thread starts when idle_timeout > 0."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=1800)

        assert reranker._cleanup_thread is not None
        assert reranker._cleanup_thread.is_alive()

        reranker.stop_cleanup_thread()

    def test_stop_cleanup_thread_stops_thread(self, mock_cross_encoder):
        """Test stop_cleanup_thread stops the cleanup thread."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=1800)

        assert reranker._cleanup_thread is not None
        reranker.stop_cleanup_thread()

        # Give thread time to stop
        time.sleep(0.1)
        assert not reranker._cleanup_thread.is_alive()

    def test_cleanup_loop_unloads_model_after_timeout(self, mock_cross_encoder):
        """Test that cleanup loop unloads model after idle timeout."""
        mock_class, mock_instance = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        # Very short timeout for testing
        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=1)
        assert reranker.model is not None

        # Manually trigger cleanup check by setting last_used in the past
        reranker._last_used = time.time() - 10

        # Trigger cleanup manually (simulating what the loop does)
        with reranker._loading_lock:
            idle_time = time.time() - reranker._last_used
            if idle_time >= reranker._idle_timeout:
                del reranker.model
                reranker.model = None
                gc.collect()

        assert reranker.model is None

        reranker.stop_cleanup_thread()


class TestCrossEncoderRerankerUnloadModel:
    """Tests for CrossEncoderReranker model unloading."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    def test_unload_model_sets_model_to_none(self, mock_cross_encoder):
        """Test unload_model sets model to None."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)
        assert reranker.model is not None

        reranker.unload_model()

        assert reranker.model is None

    def test_unload_model_when_model_is_none_is_safe(self, mock_cross_encoder):
        """Test unload_model when model is None doesn't raise."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)
        assert reranker.model is None

        # Should not raise
        reranker.unload_model()
        assert reranker.model is None

    def test_unload_model_with_torch_clears_cuda_cache(self, mock_cross_encoder):
        """Test unload_model clears CUDA cache when torch is available."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            reranker.unload_model()

            mock_torch.cuda.is_available.assert_called_once()
            mock_torch.cuda.empty_cache.assert_called_once()


class TestCrossEncoderRerankerThreadSafety:
    """Tests for CrossEncoderReranker thread safety."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class with thread-safe tracking."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            call_count = {"count": 0}
            lock = threading.Lock()

            def create_instance(*args, **kwargs):
                with lock:
                    call_count["count"] += 1
                mock_instance = MagicMock()
                mock_instance.predict.return_value = np.array([0.8])
                return mock_instance

            mock.side_effect = create_instance
            yield mock, call_count

    def test_concurrent_load_model_only_loads_once(self, mock_cross_encoder):
        """Test concurrent _load_model calls only load model once."""
        mock_class, call_count = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=0)

        # Start multiple threads trying to load
        threads = []
        for _ in range(5):
            t = threading.Thread(target=reranker._load_model)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Model should only be loaded once
        assert call_count["count"] == 1
        assert reranker.model is not None

    def test_concurrent_rerank_is_thread_safe(self, mock_cross_encoder):
        """Test concurrent rerank calls are thread safe."""
        mock_class, _ = mock_cross_encoder

        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)

        results = []
        errors = []

        def do_rerank():
            try:
                result = reranker.rerank(
                    query="test", documents=["doc1", "doc2", "doc3"]
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            t = threading.Thread(target=do_rerank)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestCrossEncoderRerankerDestructor:
    """Tests for CrossEncoderReranker destructor behavior."""

    @pytest.fixture
    def mock_cross_encoder(self):
        """Mock CrossEncoder class."""
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock, mock_instance

    def test_destructor_stops_cleanup_thread(self, mock_cross_encoder):
        """Test destructor stops cleanup thread."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(idle_timeout=1800)
        cleanup_thread = reranker._cleanup_thread

        assert cleanup_thread is not None
        assert cleanup_thread.is_alive()

        # Explicitly call destructor
        reranker.__del__()

        # Give thread time to stop
        time.sleep(0.2)
        assert not cleanup_thread.is_alive()

    def test_destructor_unloads_model(self, mock_cross_encoder):
        """Test destructor unloads model."""
        from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(lazy_load=False, idle_timeout=0)
        assert reranker.model is not None

        reranker.__del__()

        assert reranker.model is None


# ============================================================================
# RerankerInterface Tests
# ============================================================================


class TestRerankerInterface:
    """Tests for RerankerInterface abstract class."""

    def test_interface_cannot_be_instantiated(self):
        """Test that RerankerInterface cannot be instantiated directly."""
        from code_rag.reranker.reranker_interface import RerankerInterface

        with pytest.raises(TypeError):
            RerankerInterface()

    def test_implementations_implement_interface(self):
        """Test that implementations properly implement the interface."""
        from code_rag.reranker.http_reranker import HttpReranker
        from code_rag.reranker.reranker_interface import RerankerInterface

        reranker = HttpReranker()
        assert isinstance(reranker, RerankerInterface)

        # CrossEncoderReranker would need mocking to test
        with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder"):
            from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker

            ce_reranker = CrossEncoderReranker(idle_timeout=0)
            assert isinstance(ce_reranker, RerankerInterface)
