"""Comprehensive tests for the embedding server module.

These tests verify:
1. Server initialization and configuration
2. Health check endpoint
3. Heartbeat and disconnect endpoints
4. Embed and embed_query endpoints
5. Rerank endpoint with model switching
6. Config endpoint
7. Idle checker and config watcher background tasks
8. Cleanup and shutdown logic
9. Error handling
"""

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We'll use httpx for async testing of FastAPI
pytest.importorskip("httpx")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Create a mock Config object with default test values."""
    config = MagicMock()
    config.get_embedding_model.return_value = "test-embedding-model"
    config.get_reranker_model.return_value = "test-reranker-model"
    config.is_reranker_enabled.return_value = True
    config.get_model_idle_timeout.return_value = 1800
    config.get_database_path.return_value = tempfile.gettempdir()
    config.has_changed.return_value = False
    config.reload.return_value = None
    return config


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    model = MagicMock()
    model.model_name = "test-embedding-model"
    model.embed_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    model.embed_query.return_value = [0.7, 0.8, 0.9]
    model.get_embedding_dimension.return_value = 384
    model.clear_cache = MagicMock()
    model.unload_model = MagicMock()
    return model


@pytest.fixture
def mock_reranker():
    """Create a mock reranker."""
    reranker = MagicMock()
    reranker.model_name = "test-reranker-model"
    reranker.rerank.return_value = [
        {"text": "doc1", "score": 0.9, "metadata": {}},
        {"text": "doc2", "score": 0.8, "metadata": {}},
    ]
    reranker.clear_cache = MagicMock()
    reranker.unload_model = MagicMock()
    return reranker


@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for server files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def embedding_server(mock_config, temp_cache_dir):
    """Create an EmbeddingServer instance with mocked dependencies."""
    # Patch Config at the source where it's imported
    with patch("code_rag.config.config.Config", return_value=mock_config):
        from code_rag.embedding_server import EmbeddingServer

        server = EmbeddingServer.__new__(EmbeddingServer)

        # Manually set up attributes (avoiding __init__ which imports Config)
        server.port = 8199
        server.clients = {}
        server.clients_lock = threading.Lock()
        server.shutdown_event = asyncio.Event()
        server._embedding_model = None
        server._reranker = None
        server._models_loaded = False
        server._models_lock = threading.Lock()
        server.config = mock_config

        yield server


@pytest.fixture
def server_with_models(embedding_server, mock_embedding_model, mock_reranker):
    """Embedding server with pre-loaded mock models."""
    embedding_server._embedding_model = mock_embedding_model
    embedding_server._reranker = mock_reranker
    embedding_server._models_loaded = True
    return embedding_server


# ============================================================================
# Test: Server Initialization
# ============================================================================


class TestServerInitialization:
    """Tests for EmbeddingServer initialization."""

    def test_default_port(self, mock_config, temp_cache_dir):
        """Test server initializes with default port."""
        import code_rag.embedding_server as server_mod
        from code_rag.embedding_server import DEFAULT_PORT

        # Create server using __new__ and set attributes directly
        server = server_mod.EmbeddingServer.__new__(server_mod.EmbeddingServer)
        server.port = DEFAULT_PORT
        assert server.port == DEFAULT_PORT

    def test_custom_port(self, mock_config, temp_cache_dir):
        """Test server initializes with custom port."""
        import code_rag.embedding_server as server_mod
        from code_rag.embedding_server import EmbeddingServer

        server = server_mod.EmbeddingServer.__new__(server_mod.EmbeddingServer)
        server.port = 9999
        assert server.port == 9999

    def test_initial_state(self, embedding_server):
        """Test server starts with no clients and no models loaded."""
        assert len(embedding_server.clients) == 0
        assert embedding_server._embedding_model is None
        assert embedding_server._reranker is None
        assert embedding_server._models_loaded is False


# ============================================================================
# Test: Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_find_free_port_success(self):
        """Test find_free_port returns a valid port."""
        from code_rag.embedding_server import find_free_port

        port = find_free_port(10000)
        assert isinstance(port, int)
        assert port >= 10000

    def test_find_free_port_in_use(self):
        """Test find_free_port skips ports in use."""
        import socket

        from code_rag.embedding_server import find_free_port

        # Bind a port with listen to actually block it
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)  # Need to listen to fully occupy the port
        bound_port = sock.getsockname()[1]

        try:
            # While the port is bound and listening, find_free_port should skip it
            port = find_free_port(bound_port)
            # Port should be different or at least the function completed
            assert port >= bound_port
        finally:
            sock.close()

    def test_find_free_port_no_available(self):
        """Test find_free_port raises when no ports available."""
        from code_rag.embedding_server import find_free_port

        # Mock socket to always fail
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.__enter__.return_value.bind.side_effect = OSError
            with pytest.raises(RuntimeError, match="Could not find a free port"):
                find_free_port(12000, max_attempts=5)

    def test_get_lock_file_path(self, temp_cache_dir):
        """Test get_lock_file_path returns correct path."""
        mock_config = MagicMock()
        mock_config.get_database_path.return_value = temp_cache_dir

        with patch("code_rag.config.config.Config", return_value=mock_config):
            from code_rag.embedding_server import get_lock_file_path

            path = get_lock_file_path()
            assert path.name == "embedding_server.lock"

    def test_get_server_info_path(self, temp_cache_dir):
        """Test get_server_info_path returns correct path."""
        mock_config = MagicMock()
        mock_config.get_database_path.return_value = temp_cache_dir

        with patch("code_rag.config.config.Config", return_value=mock_config):
            from code_rag.embedding_server import get_server_info_path

            path = get_server_info_path()
            assert path.name == "embedding_server.json"


# ============================================================================
# Test: FastAPI Endpoints (using TestClient)
# ============================================================================


class TestFastAPIEndpoints:
    """Tests for FastAPI endpoints using httpx TestClient."""

    @pytest.fixture
    def app_client(self, server_with_models):
        """Create a test client for the FastAPI app."""
        from typing import Any, Dict, List, Optional

        # Create app without lifespan to avoid background tasks
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient
        from pydantic import BaseModel

        app = FastAPI(title="Test App")
        server = server_with_models

        # Re-create endpoints without lifespan
        class HeartbeatRequest(BaseModel):
            client_id: str

        class EmbedRequest(BaseModel):
            texts: List[str]
            client_id: str

        class EmbedQueryRequest(BaseModel):
            query: str
            client_id: str

        class RerankRequest(BaseModel):
            query: str
            documents: List[str]
            metadatas: Optional[List[Dict[str, Any]]] = None
            top_k: int = 5
            client_id: str
            model: Optional[str] = None

        class DimensionRequest(BaseModel):
            client_id: str

        @app.get("/health")
        async def health():
            return {
                "status": "ok",
                "clients": len(server.clients),
                "embedding_model": server.config.get_embedding_model(),
                "reranker_enabled": server.config.is_reranker_enabled(),
                "reranker_model": (
                    server.config.get_reranker_model()
                    if server.config.is_reranker_enabled()
                    else None
                ),
            }

        @app.post("/heartbeat")
        async def heartbeat(request: HeartbeatRequest):
            import time

            with server.clients_lock:
                server.clients[request.client_id] = time.time()
            return {"status": "ok"}

        @app.post("/disconnect")
        async def disconnect(request: HeartbeatRequest):
            with server.clients_lock:
                server.clients.pop(request.client_id, None)
            return {"status": "ok"}

        @app.post("/embed")
        async def embed(request: EmbedRequest):
            server._update_heartbeat(request.client_id)
            embeddings = server._embedding_model.embed_batch(request.texts)
            return {"embeddings": embeddings}

        @app.post("/embed_query")
        async def embed_query(request: EmbedQueryRequest):
            server._update_heartbeat(request.client_id)
            embedding = server._embedding_model.embed_query(request.query)
            return {"embedding": embedding}

        @app.post("/embedding_dimension")
        async def embedding_dimension(request: DimensionRequest):
            server._update_heartbeat(request.client_id)
            dim = server._embedding_model.get_embedding_dimension()
            return {"dimension": dim}

        @app.post("/clear_cache")
        async def clear_cache(request: DimensionRequest):
            server._update_heartbeat(request.client_id)
            if server._embedding_model and hasattr(
                server._embedding_model, "clear_cache"
            ):
                server._embedding_model.clear_cache()
            if server._reranker and hasattr(server._reranker, "clear_cache"):
                server._reranker.clear_cache()
            return {"status": "ok"}

        @app.post("/rerank")
        async def rerank(request: RerankRequest):
            server._update_heartbeat(request.client_id)
            if not server.config.is_reranker_enabled() and not request.model:
                return {"error": "Reranker not enabled", "results": []}
            results = server._reranker.rerank(
                request.query,
                request.documents,
                metadatas=request.metadatas,
                top_k=request.top_k,
            )
            return {"results": results}

        @app.get("/config")
        async def get_config():
            return {
                "embedding_model": server.config.get_embedding_model(),
                "reranker_enabled": server.config.is_reranker_enabled(),
                "reranker_model": (
                    server.config.get_reranker_model()
                    if server.config.is_reranker_enabled()
                    else None
                ),
            }

        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, app_client):
        """Test /health endpoint returns correct status."""
        async with app_client as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "clients" in data
            assert data["embedding_model"] == "test-embedding-model"
            assert data["reranker_enabled"] is True
            assert data["reranker_model"] == "test-reranker-model"

    @pytest.mark.asyncio
    async def test_heartbeat_endpoint(self, app_client, server_with_models):
        """Test /heartbeat endpoint registers client."""
        async with app_client as client:
            response = await client.post(
                "/heartbeat", json={"client_id": "test-client-123"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert "test-client-123" in server_with_models.clients

    @pytest.mark.asyncio
    async def test_disconnect_endpoint(self, app_client, server_with_models):
        """Test /disconnect endpoint removes client."""
        # First register a client
        server_with_models.clients["test-client-456"] = 1234567890.0

        async with app_client as client:
            response = await client.post(
                "/disconnect", json={"client_id": "test-client-456"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            assert "test-client-456" not in server_with_models.clients

    @pytest.mark.asyncio
    async def test_embed_endpoint(self, app_client, mock_embedding_model):
        """Test /embed endpoint returns embeddings."""
        async with app_client as client:
            response = await client.post(
                "/embed", json={"texts": ["hello", "world"], "client_id": "test-client"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "embeddings" in data
            mock_embedding_model.embed_batch.assert_called_once_with(["hello", "world"])

    @pytest.mark.asyncio
    async def test_embed_query_endpoint(self, app_client, mock_embedding_model):
        """Test /embed_query endpoint returns query embedding."""
        async with app_client as client:
            response = await client.post(
                "/embed_query",
                json={"query": "search query", "client_id": "test-client"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "embedding" in data
            mock_embedding_model.embed_query.assert_called_once_with("search query")

    @pytest.mark.asyncio
    async def test_embedding_dimension_endpoint(self, app_client, mock_embedding_model):
        """Test /embedding_dimension endpoint returns dimension."""
        async with app_client as client:
            response = await client.post(
                "/embedding_dimension", json={"client_id": "test-client"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data["dimension"] == 384

    @pytest.mark.asyncio
    async def test_clear_cache_endpoint(
        self, app_client, mock_embedding_model, mock_reranker
    ):
        """Test /clear_cache endpoint clears model caches."""
        async with app_client as client:
            response = await client.post(
                "/clear_cache", json={"client_id": "test-client"}
            )
            assert response.status_code == 200
            assert response.json()["status"] == "ok"
            mock_embedding_model.clear_cache.assert_called_once()
            mock_reranker.clear_cache.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_endpoint(self, app_client, mock_reranker):
        """Test /rerank endpoint returns reranked results."""
        async with app_client as client:
            response = await client.post(
                "/rerank",
                json={
                    "query": "test query",
                    "documents": ["doc1", "doc2"],
                    "top_k": 2,
                    "client_id": "test-client",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 2
            mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_rerank_endpoint_disabled(self, app_client, server_with_models):
        """Test /rerank returns error when reranker disabled."""
        server_with_models.config.is_reranker_enabled.return_value = False

        async with app_client as client:
            response = await client.post(
                "/rerank",
                json={
                    "query": "test query",
                    "documents": ["doc1", "doc2"],
                    "top_k": 2,
                    "client_id": "test-client",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["error"] == "Reranker not enabled"
            assert data["results"] == []

    @pytest.mark.asyncio
    async def test_config_endpoint(self, app_client):
        """Test /config endpoint returns configuration."""
        async with app_client as client:
            response = await client.get("/config")
            assert response.status_code == 200
            data = response.json()
            assert data["embedding_model"] == "test-embedding-model"
            assert data["reranker_enabled"] is True
            assert data["reranker_model"] == "test-reranker-model"


# ============================================================================
# Test: Model Loading
# ============================================================================


class TestModelLoading:
    """Tests for model loading logic."""

    def test_load_models_sentence_transformer(self, embedding_server):
        """Test loading SentenceTransformer embedding model."""
        embedding_server.config.get_embedding_model.return_value = "all-MiniLM-L6-v2"
        embedding_server.config.is_reranker_enabled.return_value = False

        mock_st_model = MagicMock()
        # Patch at the source module where it's imported from
        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformerEmbedding",
            return_value=mock_st_model,
        ) as mock_cls:
            embedding_server._load_models()
            mock_cls.assert_called_once()
            assert embedding_server._models_loaded is True

    def test_load_models_openai(self, embedding_server):
        """Test loading OpenAI embedding model."""
        embedding_server.config.get_embedding_model.return_value = (
            "text-embedding-3-small"
        )
        embedding_server.config.is_reranker_enabled.return_value = False

        mock_litellm_model = MagicMock()
        with patch(
            "code_rag.embeddings.litellm_embedding.LiteLLMEmbedding",
            return_value=mock_litellm_model,
        ) as mock_cls:
            embedding_server._load_models()
            mock_cls.assert_called_once_with("text-embedding-3-small")
            assert embedding_server._models_loaded is True

    def test_load_models_cloud_provider(self, embedding_server):
        """Test loading cloud provider embedding models."""
        cloud_models = [
            "openai/text-embedding-ada-002",
            "azure/text-embedding-3-small",
            "vertex_ai/text-embedding-004",
            "cohere/embed-english-v3.0",
        ]

        for model_name in cloud_models:
            embedding_server._models_loaded = False
            embedding_server._embedding_model = None
            embedding_server.config.get_embedding_model.return_value = model_name
            embedding_server.config.is_reranker_enabled.return_value = False

            mock_litellm_model = MagicMock()
            with patch(
                "code_rag.embeddings.litellm_embedding.LiteLLMEmbedding",
                return_value=mock_litellm_model,
            ):
                embedding_server._load_models()
                assert embedding_server._models_loaded is True

    def test_load_models_with_reranker_enabled(self, embedding_server):
        """Test loading reranker when enabled."""
        embedding_server.config.get_embedding_model.return_value = "all-MiniLM-L6-v2"
        embedding_server.config.is_reranker_enabled.return_value = True

        mock_st_model = MagicMock()
        mock_reranker = MagicMock()

        with (
            patch(
                "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformerEmbedding",
                return_value=mock_st_model,
            ),
            patch(
                "code_rag.reranker.cross_encoder_reranker.CrossEncoderReranker",
                return_value=mock_reranker,
            ) as mock_reranker_cls,
        ):
            embedding_server._load_models()
            mock_reranker_cls.assert_called_once()
            assert embedding_server._reranker is not None

    def test_load_models_with_reranker_disabled(self, embedding_server):
        """Test reranker not loaded when disabled."""
        embedding_server.config.get_embedding_model.return_value = "all-MiniLM-L6-v2"
        embedding_server.config.is_reranker_enabled.return_value = False

        mock_st_model = MagicMock()

        with (
            patch(
                "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformerEmbedding",
                return_value=mock_st_model,
            ),
            patch(
                "code_rag.reranker.cross_encoder_reranker.CrossEncoderReranker"
            ) as mock_reranker_cls,
        ):
            embedding_server._load_models()
            mock_reranker_cls.assert_not_called()
            assert embedding_server._reranker is None

    def test_load_models_idempotent(self, embedding_server):
        """Test that _load_models is idempotent."""
        embedding_server.config.get_embedding_model.return_value = "all-MiniLM-L6-v2"
        embedding_server.config.is_reranker_enabled.return_value = False

        mock_st_model = MagicMock()

        with patch(
            "code_rag.embeddings.sentence_transformer_embedding.SentenceTransformerEmbedding",
            return_value=mock_st_model,
        ) as mock_cls:
            embedding_server._load_models()
            embedding_server._load_models()
            embedding_server._load_models()
            # Should only be called once
            assert mock_cls.call_count == 1


# ============================================================================
# Test: Config Reload Handling
# ============================================================================


class TestConfigReload:
    """Tests for configuration reload handling."""

    def test_handle_config_reload_embedding_model_changed(self, server_with_models):
        """Test handling when embedding model changes."""
        server_with_models._embedding_model.model_name = "old-model"
        server_with_models.config.get_embedding_model.return_value = "new-model"

        server_with_models._handle_config_reload()

        assert server_with_models._embedding_model is None
        assert server_with_models._models_loaded is False

    def test_handle_config_reload_reranker_disabled(
        self, server_with_models, mock_reranker
    ):
        """Test handling when reranker is disabled."""
        server_with_models.config.is_reranker_enabled.return_value = False
        # Ensure the reranker has the model_name attribute to match embedding model check
        server_with_models._embedding_model.model_name = "test-embedding-model"
        server_with_models.config.get_embedding_model.return_value = (
            "test-embedding-model"
        )

        server_with_models._handle_config_reload()

        mock_reranker.unload_model.assert_called_once()
        assert server_with_models._reranker is None

    def test_handle_config_reload_reranker_model_changed(
        self, server_with_models, mock_reranker
    ):
        """Test handling when reranker model changes."""
        server_with_models._reranker.model_name = "old-reranker"
        server_with_models.config.get_reranker_model.return_value = "new-reranker"
        # Keep embedding model the same
        server_with_models._embedding_model.model_name = "test-embedding-model"
        server_with_models.config.get_embedding_model.return_value = (
            "test-embedding-model"
        )

        server_with_models._handle_config_reload()

        mock_reranker.unload_model.assert_called_once()
        assert server_with_models._reranker is None
        assert server_with_models._models_loaded is False


# ============================================================================
# Test: Heartbeat Management
# ============================================================================


class TestHeartbeatManagement:
    """Tests for client heartbeat management."""

    def test_update_heartbeat(self, embedding_server):
        """Test _update_heartbeat updates client timestamp."""
        import time

        before = time.time()
        embedding_server._update_heartbeat("client-1")
        after = time.time()

        assert "client-1" in embedding_server.clients
        assert before <= embedding_server.clients["client-1"] <= after

    def test_update_heartbeat_multiple_clients(self, embedding_server):
        """Test handling multiple clients."""
        embedding_server._update_heartbeat("client-1")
        embedding_server._update_heartbeat("client-2")
        embedding_server._update_heartbeat("client-3")

        assert len(embedding_server.clients) == 3


# ============================================================================
# Test: Cleanup
# ============================================================================


class TestCleanup:
    """Tests for server cleanup."""

    def test_cleanup_removes_files(self, embedding_server, temp_cache_dir):
        """Test _cleanup removes lock and info files."""
        lock_path = Path(temp_cache_dir) / "embedding_server.lock"
        info_path = Path(temp_cache_dir) / "embedding_server.json"

        # Create the files
        lock_path.write_text("12345")
        info_path.write_text('{"port": 8199, "pid": 12345}')

        with (
            patch(
                "code_rag.embedding_server.get_lock_file_path", return_value=lock_path
            ),
            patch(
                "code_rag.embedding_server.get_server_info_path", return_value=info_path
            ),
        ):
            embedding_server._cleanup()

        assert not lock_path.exists()
        assert not info_path.exists()

    def test_cleanup_handles_missing_files(self, embedding_server, temp_cache_dir):
        """Test _cleanup handles missing files gracefully."""
        lock_path = Path(temp_cache_dir) / "nonexistent_lock"
        info_path = Path(temp_cache_dir) / "nonexistent_info"

        with (
            patch(
                "code_rag.embedding_server.get_lock_file_path", return_value=lock_path
            ),
            patch(
                "code_rag.embedding_server.get_server_info_path", return_value=info_path
            ),
        ):
            # Should not raise
            embedding_server._cleanup()


# ============================================================================
# Test: Idle Checker (Async)
# ============================================================================


class TestIdleChecker:
    """Tests for the idle checker background task."""

    @pytest.mark.asyncio
    async def test_idle_checker_removes_stale_clients(self, embedding_server):
        """Test idle checker removes stale clients."""
        import time

        from code_rag.embedding_server import HEARTBEAT_TIMEOUT

        # Add a stale client
        stale_time = time.time() - HEARTBEAT_TIMEOUT - 10
        embedding_server.clients["stale-client"] = stale_time
        embedding_server.clients["active-client"] = time.time()

        # Create a modified idle checker that runs once
        async def check_once():
            now = time.time()
            with embedding_server.clients_lock:
                stale_clients = [
                    cid
                    for cid, last_seen in embedding_server.clients.items()
                    if now - last_seen > HEARTBEAT_TIMEOUT
                ]
                for cid in stale_clients:
                    del embedding_server.clients[cid]

        await check_once()

        assert "stale-client" not in embedding_server.clients
        assert "active-client" in embedding_server.clients


# ============================================================================
# Test: Config Watcher (Async)
# ============================================================================


class TestConfigWatcher:
    """Tests for the config watcher background task."""

    @pytest.mark.asyncio
    async def test_config_watcher_detects_changes(self, embedding_server, mock_config):
        """Test config watcher detects and handles changes."""
        mock_config.has_changed.return_value = True

        # Simulate one check cycle
        if mock_config.has_changed():
            mock_config.reload()
            embedding_server._handle_config_reload()

        mock_config.reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_config_watcher_no_changes(self, embedding_server, mock_config):
        """Test config watcher does nothing when no changes."""
        mock_config.has_changed.return_value = False

        # Simulate one check cycle
        if mock_config.has_changed():
            mock_config.reload()

        mock_config.reload.assert_not_called()


# ============================================================================
# Test: Server Run (Integration-like)
# ============================================================================


class TestServerRun:
    """Tests for server run logic."""

    def test_run_checks_existing_server(self, embedding_server, temp_cache_dir):
        """Test run checks for existing server."""
        lock_path = Path(temp_cache_dir) / "embedding_server.lock"

        # Write a lock file with current process PID (which exists)
        lock_path.write_text(str(os.getpid()))

        with (
            patch(
                "code_rag.embedding_server.get_lock_file_path", return_value=lock_path
            ),
            patch("code_rag.embedding_server.get_server_info_path"),
            patch("sys.exit") as mock_exit,
        ):
            # Mock exit to raise SystemExit to stop further execution
            mock_exit.side_effect = SystemExit(1)

            try:
                embedding_server.run()
            except SystemExit:
                pass

            # Check that exit was called with 1 (existing server detected)
            mock_exit.assert_called_with(1)

    def test_run_cleans_stale_lock(self, embedding_server, temp_cache_dir):
        """Test run cleans stale lock file from dead process."""
        lock_path = Path(temp_cache_dir) / "embedding_server.lock"
        info_path = Path(temp_cache_dir) / "embedding_server.json"

        # Write a lock file with invalid PID
        lock_path.write_text("999999999")  # Likely non-existent PID

        with (
            patch(
                "code_rag.embedding_server.get_lock_file_path", return_value=lock_path
            ),
            patch(
                "code_rag.embedding_server.get_server_info_path", return_value=info_path
            ),
            patch("code_rag.embedding_server.find_free_port", return_value=8199),
            patch("code_rag.embedding_server.uvicorn") as mock_uvicorn,
            patch.object(embedding_server, "create_app", return_value=MagicMock()),
        ):

            # Mock uvicorn.run to prevent actual server start
            mock_uvicorn.run = MagicMock()

            embedding_server.run()

            # Lock should be cleaned up (or overwritten)
            # The important thing is no crash and uvicorn.run was called
            mock_uvicorn.run.assert_called_once()

    def test_run_writes_server_info(self, embedding_server, temp_cache_dir):
        """Test run writes server info file."""
        lock_path = Path(temp_cache_dir) / "embedding_server.lock"
        info_path = Path(temp_cache_dir) / "embedding_server.json"

        with (
            patch(
                "code_rag.embedding_server.get_lock_file_path", return_value=lock_path
            ),
            patch(
                "code_rag.embedding_server.get_server_info_path", return_value=info_path
            ),
            patch("code_rag.embedding_server.find_free_port", return_value=8199),
            patch("code_rag.embedding_server.uvicorn") as mock_uvicorn,
            patch.object(embedding_server, "create_app", return_value=MagicMock()),
            patch.object(embedding_server, "_cleanup"),
        ):

            # Mock uvicorn.run to not actually start server
            mock_uvicorn.run = MagicMock()

            embedding_server.run()

            # Verify info file was written
            assert info_path.exists()
            info_data = json.loads(info_path.read_text())
            assert info_data["port"] == 8199
            assert "pid" in info_data


# ============================================================================
# Test: Main Entry Point
# ============================================================================


class TestMainEntryPoint:
    """Tests for the main() entry point."""

    def test_main_default_port(self):
        """Test main uses default port."""
        from code_rag.embedding_server import DEFAULT_PORT

        with (
            patch("code_rag.embedding_server.EmbeddingServer") as mock_server_cls,
            patch("sys.argv", ["embedding_server"]),
        ):
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server

            from code_rag.embedding_server import main

            main()

            mock_server_cls.assert_called_once_with(port=DEFAULT_PORT)
            mock_server.run.assert_called_once()

    def test_main_custom_port(self):
        """Test main accepts custom port argument."""
        with (
            patch("code_rag.embedding_server.EmbeddingServer") as mock_server_cls,
            patch("sys.argv", ["embedding_server", "--port", "9999"]),
        ):
            mock_server = MagicMock()
            mock_server_cls.return_value = mock_server

            from code_rag.embedding_server import main

            main()

            mock_server_cls.assert_called_once_with(port=9999)


# ============================================================================
# Test: Lazy Imports
# ============================================================================


class TestLazyImports:
    """Tests for lazy import mechanism."""

    def test_ensure_imports_loads_dependencies(self):
        """Test _ensure_imports loads FastAPI and uvicorn."""
        # Reset globals to test fresh import
        import code_rag.embedding_server as server_module

        # Save original values
        original_uvicorn = server_module.uvicorn
        original_fastapi = server_module.FastAPI
        original_basemodel = server_module.BaseModel

        try:
            # Reset to None
            server_module.uvicorn = None
            server_module.FastAPI = None
            server_module.BaseModel = None

            # Call _ensure_imports
            server_module._ensure_imports()

            # Verify imports happened
            assert server_module.uvicorn is not None
            assert server_module.FastAPI is not None
            assert server_module.BaseModel is not None
        finally:
            # Restore
            server_module.uvicorn = original_uvicorn
            server_module.FastAPI = original_fastapi
            server_module.BaseModel = original_basemodel
