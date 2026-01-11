"""Comprehensive tests for the embedding modules.

These tests verify:
1. HTTPEmbedding server management and request handling
2. LiteLLMEmbedding initialization and embedding generation
3. OpenAIEmbedding initialization and embedding generation
4. SentenceTransformerEmbedding model loading, caching, and idle timeout

All tests mock external dependencies to avoid real HTTP requests and model loading.
"""

import gc
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock, mock_open, patch

import numpy as np
import pytest
import requests

# ============================================================================
# HTTPEmbedding Tests
# ============================================================================


class TestHttpEmbeddingInit:
    """Tests for HTTPEmbedding initialization."""

    @pytest.fixture
    def mock_server_running(self):
        """Mock a running server."""
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            # Health check returns 200
            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)
            yield mock_get, mock_post

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide a temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_init_connects_to_running_server(self, mock_server_running, temp_cache_dir):
        """Test that initialization connects to an already running server."""
        mock_get, mock_post = mock_server_running

        with (
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):
            # Server info file doesn't exist
            mock_info_path.return_value = Path(temp_cache_dir) / "embedding_server.json"
            mock_log_path.return_value = Path(temp_cache_dir) / "embedding_server.log"

            from code_rag.embeddings.http_embedding import HttpEmbedding

            embedding = HttpEmbedding(port=8199)

            assert embedding.port == 8199
            assert embedding.base_url == "http://127.0.0.1:8199"
            assert embedding.client_id is not None

            # Cleanup
            embedding._cleanup()

    def test_init_reads_port_from_server_info(
        self, mock_server_running, temp_cache_dir
    ):
        """Test that initialization reads actual port from server info file."""
        mock_get, mock_post = mock_server_running

        # Create server info file with different port
        info_path = Path(temp_cache_dir) / "embedding_server.json"
        info_path.write_text(json.dumps({"port": 9999}))

        with (
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):
            mock_info_path.return_value = info_path
            mock_log_path.return_value = Path(temp_cache_dir) / "embedding_server.log"

            from code_rag.embeddings.http_embedding import HttpEmbedding

            embedding = HttpEmbedding(port=8199)

            # Should have switched to the port from the info file
            assert embedding.port == 9999
            assert embedding.base_url == "http://127.0.0.1:9999"

            # Cleanup
            embedding._cleanup()


class TestHttpEmbeddingServerManagement:
    """Tests for HTTPEmbedding server spawning and health checks."""

    def test_is_server_running_returns_true_when_healthy(self):
        """Test _is_server_running returns True when server responds with 200."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                assert embedding._is_server_running() is True
                embedding._cleanup()

    def test_is_server_running_returns_false_on_connection_error(self):
        """Test _is_server_running returns False when connection fails."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
            patch.object(
                __import__(
                    "code_rag.embeddings.http_embedding", fromlist=["HttpEmbedding"]
                ).HttpEmbedding,
                "_spawn_server",
            ) as mock_spawn,
        ):

            # First call succeeds (init), subsequent calls fail
            mock_get.side_effect = [Mock(status_code=200), requests.ConnectionError()]
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                assert embedding._is_server_running() is False
                embedding._cleanup()

    def test_is_server_running_returns_false_on_timeout(self):
        """Test _is_server_running returns False when request times out."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            # First call succeeds (init), subsequent calls timeout
            mock_get.side_effect = [Mock(status_code=200), requests.Timeout()]
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                assert embedding._is_server_running() is False
                embedding._cleanup()


class TestHttpEmbeddingHeartbeat:
    """Tests for HTTPEmbedding heartbeat functionality."""

    def test_heartbeat_thread_starts(self):
        """Test that heartbeat thread is started on initialization."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                assert embedding._heartbeat_thread is not None
                assert embedding._heartbeat_thread.is_alive()

                embedding._cleanup()

    def test_heartbeat_stops_on_cleanup(self):
        """Test that heartbeat thread stops on cleanup."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                heartbeat_thread = embedding._heartbeat_thread
                embedding._cleanup()

                # Give thread time to stop
                time.sleep(0.1)
                assert not heartbeat_thread.is_alive()


class TestHttpEmbeddingRequests:
    """Tests for HTTPEmbedding embedding requests."""

    @pytest.fixture
    def mock_http_embedding(self):
        """Provide a mocked HTTPEmbedding instance."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200, json=lambda: {})

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                yield embedding, mock_post

                embedding._cleanup()

    def test_embed_single_text(self, mock_http_embedding):
        """Test embedding a single text."""
        embedding, mock_post = mock_http_embedding

        expected_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"embeddings": [expected_embedding]}
        )
        mock_post.return_value.raise_for_status = Mock()

        result = embedding.embed("test text")

        assert result == expected_embedding

    def test_embed_query(self, mock_http_embedding):
        """Test embedding a query."""
        embedding, mock_post = mock_http_embedding

        expected_embedding = [0.5, 0.4, 0.3, 0.2, 0.1]
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"embedding": expected_embedding}
        )
        mock_post.return_value.raise_for_status = Mock()

        result = embedding.embed_query("search query")

        assert result == expected_embedding

    def test_embed_batch(self, mock_http_embedding):
        """Test embedding multiple texts in batch."""
        embedding, mock_post = mock_http_embedding

        expected_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_post.return_value = Mock(
            status_code=200, json=lambda: {"embeddings": expected_embeddings}
        )
        mock_post.return_value.raise_for_status = Mock()

        result = embedding.embed_batch(["text1", "text2", "text3"])

        assert result == expected_embeddings

    def test_get_embedding_dimension(self, mock_http_embedding):
        """Test getting embedding dimension."""
        embedding, mock_post = mock_http_embedding

        mock_post.return_value = Mock(status_code=200, json=lambda: {"dimension": 384})
        mock_post.return_value.raise_for_status = Mock()

        result = embedding.get_embedding_dimension()

        assert result == 384


class TestHttpEmbeddingRetry:
    """Tests for HTTPEmbedding retry logic."""

    def test_request_retries_on_connection_error(self):
        """Test that requests retry once on connection error."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            # Health checks succeed
            mock_get.return_value = Mock(status_code=200)

            success_response = Mock(status_code=200)
            success_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
            success_response.raise_for_status = Mock()

            # Initial mock for server initialization
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                # Now set up retry scenario using a callable side_effect
                call_count = [0]

                def post_side_effect(*args, **kwargs):
                    url = args[0] if args else kwargs.get("url", "")
                    # Allow heartbeat and disconnect calls to succeed
                    if "/heartbeat" in url or "/disconnect" in url:
                        return Mock(status_code=200)
                    # First embed call fails, retry succeeds
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise requests.ConnectionError("First attempt failed")
                    return success_response

                mock_post.side_effect = post_side_effect

                result = embedding.embed("test")
                assert result == [0.1, 0.2, 0.3]

                embedding._cleanup()

    def test_request_fails_after_retry(self):
        """Test that requests fail if retry also fails."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                # Use callable side_effect that fails for embed calls but allows cleanup
                def post_side_effect(*args, **kwargs):
                    url = args[0] if args else kwargs.get("url", "")
                    if "/heartbeat" in url or "/disconnect" in url:
                        return Mock(status_code=200)
                    # All embed calls fail
                    raise requests.ConnectionError("Connection failed")

                mock_post.side_effect = post_side_effect

                with pytest.raises(RuntimeError, match="Failed to connect"):
                    embedding.embed("test")

                embedding._cleanup()


class TestHttpEmbeddingNoOpMethods:
    """Tests for HTTPEmbedding no-op methods."""

    def test_start_background_loading_is_noop(self):
        """Test that start_background_loading does nothing."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                # Should not raise
                embedding.start_background_loading()

                embedding._cleanup()

    def test_unload_model_is_noop(self):
        """Test that unload_model does nothing."""
        with (
            patch("requests.get") as mock_get,
            patch("requests.post") as mock_post,
            patch(
                "code_rag.embeddings.http_embedding.get_server_info_path"
            ) as mock_info_path,
            patch(
                "code_rag.embeddings.http_embedding.get_server_log_path"
            ) as mock_log_path,
        ):

            mock_get.return_value = Mock(status_code=200)
            mock_post.return_value = Mock(status_code=200)

            with tempfile.TemporaryDirectory() as temp_dir:
                mock_info_path.return_value = Path(temp_dir) / "embedding_server.json"
                mock_log_path.return_value = Path(temp_dir) / "embedding_server.log"

                from code_rag.embeddings.http_embedding import HttpEmbedding

                embedding = HttpEmbedding()

                # Should not raise
                embedding.unload_model()

                embedding._cleanup()


# ============================================================================
# LiteLLMEmbedding Tests
# ============================================================================


class TestLiteLLMEmbeddingInit:
    """Tests for LiteLLMEmbedding initialization."""

    def test_init_with_model_name(self):
        """Test initialization with model name."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")

        assert embedding.model_name == "text-embedding-3-small"
        assert embedding.api_key is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(
            model_name="text-embedding-3-small", api_key="test-api-key"
        )

        assert embedding.model_name == "text-embedding-3-small"
        assert embedding.api_key == "test-api-key"

    def test_init_with_idle_timeout(self):
        """Test initialization stores idle_timeout (even though not used)."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(
            model_name="text-embedding-3-small", idle_timeout=3600
        )

        assert embedding._idle_timeout == 3600


class TestLiteLLMEmbeddingEmbed:
    """Tests for LiteLLMEmbedding embedding methods."""

    @pytest.fixture
    def mock_litellm_embedding(self):
        """Provide a mocked LiteLLM embedding response."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]

        with patch("code_rag.embeddings.litellm_embedding.embedding") as mock_embed:
            mock_embed.return_value = mock_response
            yield mock_embed

    def test_embed_single_text(self, mock_litellm_embedding):
        """Test embedding a single text."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")
        result = embedding.embed("test text")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_litellm_embedding.assert_called_once()

    def test_embed_replaces_newlines(self, mock_litellm_embedding):
        """Test that embed replaces newlines in text."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")
        embedding.embed("test\ntext\nwith\nnewlines")

        # Verify newlines were replaced
        call_args = mock_litellm_embedding.call_args
        assert "\n" not in call_args.kwargs["input"][0]
        assert "test text with newlines" == call_args.kwargs["input"][0]

    def test_embed_batch(self, mock_litellm_embedding):
        """Test embedding multiple texts in batch."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9]),
        ]
        mock_litellm_embedding.return_value = mock_response

        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")
        result = embedding.embed_batch(["text1", "text2", "text3"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    def test_embed_batch_replaces_newlines(self, mock_litellm_embedding):
        """Test that embed_batch replaces newlines in all texts."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2])]
        mock_litellm_embedding.return_value = mock_response

        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")
        embedding.embed_batch(["text\nwith\nnewlines"])

        call_args = mock_litellm_embedding.call_args
        assert "text with newlines" == call_args.kwargs["input"][0]


class TestLiteLLMEmbeddingDimension:
    """Tests for LiteLLMEmbedding get_embedding_dimension."""

    def test_get_dimension_known_model(self):
        """Test getting dimension for known models."""
        from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

        test_cases = [
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
            ("vertex_ai/text-embedding-004", 768),
        ]

        for model_name, expected_dim in test_cases:
            embedding = LiteLLMEmbedding(model_name=model_name)
            assert embedding.get_embedding_dimension() == expected_dim

    def test_get_dimension_unknown_model_fallback(self):
        """Test getting dimension for unknown model via API call."""
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 512)]

        with patch("code_rag.embeddings.litellm_embedding.embedding") as mock_embed:
            mock_embed.return_value = mock_response

            from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

            embedding = LiteLLMEmbedding(model_name="unknown-model")
            assert embedding.get_embedding_dimension() == 512

    def test_get_dimension_unknown_model_error_fallback(self):
        """Test getting dimension falls back to 1536 on error."""
        with patch("code_rag.embeddings.litellm_embedding.embedding") as mock_embed:
            mock_embed.side_effect = Exception("API error")

            from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

            embedding = LiteLLMEmbedding(model_name="unknown-model")
            assert embedding.get_embedding_dimension() == 1536


# ============================================================================
# OpenAIEmbedding Tests
# ============================================================================


class TestOpenAIEmbeddingInit:
    """Tests for OpenAIEmbedding initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding(api_key="test-api-key")

            assert embedding.model_name == "text-embedding-3-small"
            mock_openai.assert_called_once_with(api_key="test-api-key")

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding(
                model_name="text-embedding-3-large", api_key="test-api-key"
            )

            assert embedding.model_name == "text-embedding-3-large"

    def test_init_uses_env_var_when_no_key(self):
        """Test initialization uses OPENAI_API_KEY env var when no key provided."""
        with (
            patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai,
            patch.dict("os.environ", {"OPENAI_API_KEY": "env-api-key"}),
        ):
            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding()

            # Should pass the env var to OpenAI
            mock_openai.assert_called_once_with(api_key="env-api-key")


class TestOpenAIEmbeddingEmbed:
    """Tests for OpenAIEmbedding embedding methods."""

    @pytest.fixture
    def mock_openai_client(self):
        """Provide a mocked OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])]
        mock_client.embeddings.create.return_value = mock_response

        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client
            yield mock_client

    def test_embed_single_text(self, mock_openai_client):
        """Test embedding a single text."""
        from code_rag.embeddings.openai_embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test-key")
        result = embedding.embed("test text")

        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_openai_client.embeddings.create.assert_called_once()

    def test_embed_replaces_newlines(self, mock_openai_client):
        """Test that embed replaces newlines in text."""
        from code_rag.embeddings.openai_embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test-key")
        embedding.embed("test\ntext\nwith\nnewlines")

        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == ["test text with newlines"]

    def test_embed_batch(self, mock_openai_client):
        """Test embedding multiple texts in batch."""
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
            MagicMock(embedding=[0.7, 0.8, 0.9]),
        ]
        mock_openai_client.embeddings.create.return_value = mock_response

        from code_rag.embeddings.openai_embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test-key")
        result = embedding.embed_batch(["text1", "text2", "text3"])

        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

    def test_embed_batch_replaces_newlines(self, mock_openai_client):
        """Test that embed_batch replaces newlines in all texts."""
        from code_rag.embeddings.openai_embedding import OpenAIEmbedding

        embedding = OpenAIEmbedding(api_key="test-key")
        embedding.embed_batch(["text\none", "text\ntwo"])

        call_args = mock_openai_client.embeddings.create.call_args
        assert call_args.kwargs["input"] == ["text one", "text two"]


class TestOpenAIEmbeddingDimension:
    """Tests for OpenAIEmbedding get_embedding_dimension."""

    def test_get_dimension_known_model(self):
        """Test getting dimension for known models."""
        with patch("code_rag.embeddings.openai_embedding.OpenAI"):
            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            test_cases = [
                ("text-embedding-3-small", 1536),
                ("text-embedding-3-large", 3072),
                ("text-embedding-ada-002", 1536),
            ]

            for model_name, expected_dim in test_cases:
                embedding = OpenAIEmbedding(model_name=model_name, api_key="test-key")
                assert embedding.get_embedding_dimension() == expected_dim

    def test_get_dimension_unknown_model_fallback(self):
        """Test getting dimension for unknown model via API call."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 768)]
        mock_client.embeddings.create.return_value = mock_response

        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding(model_name="unknown-model", api_key="test-key")
            assert embedding.get_embedding_dimension() == 768

    def test_get_dimension_unknown_model_error_fallback(self):
        """Test getting dimension falls back to 1536 on error."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API error")

        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding(model_name="unknown-model", api_key="test-key")
            assert embedding.get_embedding_dimension() == 1536


# ============================================================================
# SentenceTransformerEmbedding Tests
# ============================================================================


class TestSentenceTransformerEmbeddingInit:
    """Tests for SentenceTransformerEmbedding initialization."""

    def test_init_default_model(self, mock_sentence_transformer_model):
        """Test initialization with default model."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()

            assert embedding.model_name == "all-MiniLM-L6-v2"
            assert embedding.model is not None
            embedding.stop_cleanup_thread()

    def test_init_custom_model(self, mock_sentence_transformer_model):
        """Test initialization with custom model name."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(model_name="custom-model")

            assert embedding.model_name == "custom-model"
            embedding.stop_cleanup_thread()

    def test_init_lazy_load(self, mock_sentence_transformer_model):
        """Test initialization with lazy loading."""
        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer"
        ) as mock_st:
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(lazy_load=True)

            # Model should not be loaded yet
            assert embedding.model is None
            mock_st.assert_not_called()
            embedding.stop_cleanup_thread()

    def test_init_query_prefix_for_code_rank_embed(
        self, mock_sentence_transformer_model
    ):
        """Test that CodeRankEmbed model gets query prefix."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(
                model_name="nomic-ai/CodeRankEmbed"
            )

            assert (
                embedding.query_prefix
                == "Represent this query for searching relevant code: "
            )
            embedding.stop_cleanup_thread()


class TestSentenceTransformerEmbeddingEmbed:
    """Tests for SentenceTransformerEmbedding embedding methods."""

    def test_embed_single_text(self, mock_sentence_transformer_model):
        """Test embedding a single text."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()
            result = embedding.embed("test text")

            assert isinstance(result, list)
            assert len(result) == 384
            assert all(isinstance(x, float) for x in result)
            embedding.stop_cleanup_thread()

    def test_embed_query_with_prefix(self, mock_sentence_transformer_model):
        """Test embedding a query with model-specific prefix."""
        mock_model = mock_sentence_transformer_model()

        def create_mock(*args, **kwargs):
            return mock_model

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(
                model_name="nomic-ai/CodeRankEmbed"
            )
            embedding.embed_query("search query")

            # The query should have been prefixed
            # We can verify by checking the mock was called
            assert embedding.query_prefix != ""
            embedding.stop_cleanup_thread()

    def test_embed_batch(self, mock_sentence_transformer_model):
        """Test embedding multiple texts in batch."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()
            result = embedding.embed_batch(["text1", "text2", "text3"])

            assert isinstance(result, list)
            assert len(result) == 3
            for vec in result:
                assert isinstance(vec, list)
                assert len(vec) == 384
            embedding.stop_cleanup_thread()

    def test_get_embedding_dimension(self, mock_sentence_transformer_model):
        """Test getting embedding dimension."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(embedding_dim=512)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()
            dim = embedding.get_embedding_dimension()

            assert dim == 512
            embedding.stop_cleanup_thread()


class TestSentenceTransformerEmbeddingLazyLoad:
    """Tests for SentenceTransformerEmbedding lazy loading behavior."""

    def test_lazy_load_defers_model_loading(self, mock_sentence_transformer_model):
        """Test that lazy_load=True defers model loading."""
        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer"
        ) as mock_st:
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(lazy_load=True)

            assert embedding.model is None
            mock_st.assert_not_called()
            embedding.stop_cleanup_thread()

    def test_embed_triggers_model_load(self, mock_sentence_transformer_model):
        """Test that embed() triggers model loading when lazy."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ) as mock_st:
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(lazy_load=True)

            assert embedding.model is None
            embedding.embed("trigger load")
            assert embedding.model is not None
            embedding.stop_cleanup_thread()

    def test_background_loading(self, mock_sentence_transformer_model):
        """Test background model loading."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model(load_delay=0.1)

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(lazy_load=True)
            embedding.start_background_loading()

            # Give background thread time to complete
            time.sleep(0.2)

            assert embedding.model is not None
            embedding.stop_cleanup_thread()


class TestSentenceTransformerEmbeddingIdleTimeout:
    """Tests for SentenceTransformerEmbedding idle timeout behavior."""

    def test_cleanup_thread_starts(self, mock_sentence_transformer_model):
        """Test that cleanup thread starts when idle_timeout > 0."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(idle_timeout=60)

            assert embedding._cleanup_thread is not None
            assert embedding._cleanup_thread.is_alive()
            embedding.stop_cleanup_thread()

    def test_no_cleanup_thread_when_timeout_zero(self, mock_sentence_transformer_model):
        """Test that no cleanup thread starts when idle_timeout = 0."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding(idle_timeout=0)

            # Cleanup thread should not have started
            # (it might be None or not alive)
            if embedding._cleanup_thread is not None:
                assert not embedding._cleanup_thread.is_alive()

    def test_unload_model(self, mock_sentence_transformer_model):
        """Test explicit model unloading."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()

            assert embedding.model is not None
            embedding.unload_model()
            assert embedding.model is None
            embedding.stop_cleanup_thread()

    def test_clear_cache_calls_gc(self, mock_sentence_transformer_model):
        """Test that clear_cache triggers garbage collection."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with (
            patch(
                "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
                side_effect=create_mock,
            ),
            patch(
                "code_rag.embeddings.sentence_transformer_embedding.gc.collect"
            ) as mock_gc,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()
            embedding.clear_cache()

            mock_gc.assert_called()
            embedding.stop_cleanup_thread()


class TestSentenceTransformerEmbeddingThreadSafety:
    """Tests for SentenceTransformerEmbedding thread safety."""

    def test_concurrent_embed_calls(self, mock_sentence_transformer_model):
        """Test that concurrent embed calls are thread-safe."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()

            results = []
            errors = []

            def embed_task(text):
                try:
                    result = embedding.embed(text)
                    results.append(result)
                except Exception as e:
                    errors.append(e)

            threads = [
                threading.Thread(target=embed_task, args=(f"text{i}",))
                for i in range(10)
            ]

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(results) == 10
            embedding.stop_cleanup_thread()

    def test_last_used_timestamp_updated(self, mock_sentence_transformer_model):
        """Test that _last_used is updated on embed calls."""

        def create_mock(*args, **kwargs):
            return mock_sentence_transformer_model()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer",
            side_effect=create_mock,
        ):
            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            embedding = SentenceTransformerEmbedding()

            initial_time = embedding._last_used
            time.sleep(0.1)
            embedding.embed("test")
            updated_time = embedding._last_used

            assert updated_time > initial_time
            embedding.stop_cleanup_thread()


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEmbeddingErrorHandling:
    """Tests for error handling across embedding implementations."""

    def test_litellm_api_error(self):
        """Test LiteLLM handles API errors gracefully."""
        with patch("code_rag.embeddings.litellm_embedding.embedding") as mock_embed:
            mock_embed.side_effect = Exception("API rate limit exceeded")

            from code_rag.embeddings.litellm_embedding import LiteLLMEmbedding

            embedding = LiteLLMEmbedding(model_name="text-embedding-3-small")

            with pytest.raises(Exception, match="API rate limit exceeded"):
                embedding.embed("test")

    def test_openai_api_error(self):
        """Test OpenAI handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("Invalid API key")

        with patch("code_rag.embeddings.openai_embedding.OpenAI") as mock_openai:
            mock_openai.return_value = mock_client

            from code_rag.embeddings.openai_embedding import OpenAIEmbedding

            embedding = OpenAIEmbedding(api_key="invalid-key")

            with pytest.raises(Exception, match="Invalid API key"):
                embedding.embed("test")

    def test_sentence_transformer_model_load_error(self):
        """Test SentenceTransformer handles model loading errors."""
        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformer"
        ) as mock_st:
            mock_st.side_effect = Exception("Model not found")

            from code_rag.embeddings.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            with pytest.raises(Exception, match="Model not found"):
                SentenceTransformerEmbedding(model_name="nonexistent-model")
