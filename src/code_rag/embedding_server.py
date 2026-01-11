"""On-demand embedding server for shared model access.

This server loads transformer models once and serves multiple MCP clients,
avoiding duplicate model instances that consume excessive memory.

Design:
- Auto-spawns when first client needs it
- Auto-terminates when no clients remain (after idle timeout)
- Uses lock file to prevent multiple server instances
- Tracks clients via heartbeat mechanism
"""

import asyncio
import json
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy imports to keep startup fast
uvicorn = None
FastAPI = None
BaseModel = None


def _ensure_imports():
    """Lazy import heavy dependencies."""
    global uvicorn, FastAPI, BaseModel
    if FastAPI is None:
        import uvicorn as _uvicorn
        from fastapi import FastAPI as _FastAPI
        from pydantic import BaseModel as _BaseModel

        uvicorn = _uvicorn
        FastAPI = _FastAPI
        BaseModel = _BaseModel


# Server configuration
DEFAULT_PORT = 8199
HEARTBEAT_TIMEOUT = 120  # 2 minutes - client considered dead if no heartbeat
IDLE_SHUTDOWN_DELAY = 30  # Additional grace period after last client leaves
CHECK_INTERVAL = 30  # How often to check for idle state


def get_lock_file_path() -> Path:
    """Get the path to the server lock file."""
    from .config.config import Config

    config = Config()
    cache_dir = Path(config.get_database_path())
    return cache_dir / "embedding_server.lock"


def get_server_info_path() -> Path:
    """Get the path to the server info file (contains port, pid)."""
    from .config.config import Config

    config = Config()
    cache_dir = Path(config.get_database_path())
    return cache_dir / "embedding_server.json"


def find_free_port(start_port: int, max_attempts: int = 100) -> int:
    """Find a free port starting from start_port.

    Args:
        start_port: Port number to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        An available port number

    Raises:
        RuntimeError: If no free port is found
    """
    import socket

    for offset in range(max_attempts):
        port = start_port + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue

    raise RuntimeError(
        f"Could not find a free port in range {start_port}-{start_port + max_attempts}"
    )


class EmbeddingServer:
    """HTTP server that loads models once and serves multiple clients."""

    def __init__(self, port: int = DEFAULT_PORT):
        _ensure_imports()

        self.port = port
        self.clients: Dict[str, float] = {}  # client_id -> last_heartbeat_time
        self.clients_lock = threading.Lock()
        self.shutdown_event = asyncio.Event()

        # Models (loaded lazily on first request)
        self._embedding_model = None
        self._reranker = None
        self._models_loaded = False
        self._models_lock = threading.Lock()

        # Config
        from .config.config import Config

        self.config = Config()

    def _load_models(self):
        """Load embedding and reranker models."""
        if self._models_loaded:
            return

        with self._models_lock:
            if self._models_loaded:
                return

            # Load embedding model if not already loaded
            if self._embedding_model is None:
                print(
                    f"Loading embedding model: {self.config.get_embedding_model()}",
                    file=sys.stderr,
                )
                model_name = self.config.get_embedding_model()
                # Known cloud provider prefixes supported by LiteLLM
                cloud_prefixes = [
                    "openai/",
                    "azure/",
                    "vertex_ai/",
                    "bedrock/",
                    "cohere/",
                    "voyage/",
                    "mistral/",
                    "anthropic/",
                    "huggingface/",
                    "replicate/",
                    "together_ai/",
                ]

                if model_name.startswith("text-embedding-") or any(
                    model_name.startswith(prefix) for prefix in cloud_prefixes
                ):
                    from .embeddings.litellm_embedding import LiteLLMEmbedding

                    self._embedding_model = LiteLLMEmbedding(model_name)
                else:
                    from .embeddings.sentence_transformer_embedding import (
                        SentenceTransformerEmbedding,
                    )

                    # Use configured idle timeout (default 30 min) to release VRAM
                    idle_timeout = self.config.get_model_idle_timeout()
                    self._embedding_model = SentenceTransformerEmbedding(
                        model_name, lazy_load=False, idle_timeout=idle_timeout
                    )

            # Load reranker if enabled and not already loaded
            if self.config.is_reranker_enabled() and self._reranker is None:
                print(
                    f"Loading reranker model: {self.config.get_reranker_model()}",
                    file=sys.stderr,
                )
                from .reranker.cross_encoder_reranker import CrossEncoderReranker

                idle_timeout = self.config.get_model_idle_timeout()
                self._reranker = CrossEncoderReranker(
                    self.config.get_reranker_model(),
                    lazy_load=False,
                    idle_timeout=idle_timeout,
                )

            self._models_loaded = True
            print("Models prepared successfully", file=sys.stderr)

    def create_app(self):
        """Create the FastAPI application."""
        _ensure_imports()

        @asynccontextmanager
        async def lifespan(app):
            # Startup: start idle checker and config watcher
            checker_task = asyncio.create_task(self._idle_checker())
            config_watcher_task = asyncio.create_task(self._config_watcher())
            yield
            # Shutdown
            checker_task.cancel()
            config_watcher_task.cancel()
            try:
                await asyncio.gather(checker_task, config_watcher_task)
            except asyncio.CancelledError:
                pass

        app = FastAPI(title="Code-RAG Embedding Server", lifespan=lifespan)

        # Define Pydantic models for request/response
        from pydantic import BaseModel as PydanticBaseModel

        class HeartbeatRequest(PydanticBaseModel):
            client_id: str

        class EmbedRequest(PydanticBaseModel):
            texts: List[str]
            client_id: str

        class EmbedQueryRequest(PydanticBaseModel):
            query: str
            client_id: str

        class RerankRequest(PydanticBaseModel):
            query: str
            documents: List[str]
            metadatas: Optional[List[Dict[str, Any]]] = None
            top_k: int = 5
            client_id: str
            model: Optional[str] = None

        class DimensionRequest(PydanticBaseModel):
            client_id: str

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "ok",
                "clients": len(self.clients),
                "embedding_model": self.config.get_embedding_model(),
                "reranker_enabled": self.config.is_reranker_enabled(),
                "reranker_model": (
                    self.config.get_reranker_model()
                    if self.config.is_reranker_enabled()
                    else None
                ),
            }

        @app.post("/heartbeat")
        async def heartbeat(request: HeartbeatRequest):
            """Register/update client heartbeat."""
            with self.clients_lock:
                self.clients[request.client_id] = time.time()
            return {"status": "ok"}

        @app.post("/disconnect")
        async def disconnect(request: HeartbeatRequest):
            """Client disconnect notification."""
            with self.clients_lock:
                self.clients.pop(request.client_id, None)
            return {"status": "ok"}

        @app.post("/embed")
        async def embed(request: EmbedRequest):
            """Embed a batch of texts."""
            self._update_heartbeat(request.client_id)
            self._load_models()

            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self._embedding_model.embed_batch, request.texts
            )
            return {"embeddings": embeddings}

        @app.post("/embed_query")
        async def embed_query(request: EmbedQueryRequest):
            """Embed a query with model-specific prefix."""
            self._update_heartbeat(request.client_id)
            self._load_models()

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self._embedding_model.embed_query, request.query
            )
            return {"embedding": embedding}

        @app.post("/embedding_dimension")
        async def embedding_dimension(request: DimensionRequest):
            """Get embedding dimension."""
            self._update_heartbeat(request.client_id)
            self._load_models()

            dim = self._embedding_model.get_embedding_dimension()
            return {"dimension": dim}

        @app.post("/clear_cache")
        async def clear_cache(request: DimensionRequest):
            """Clear memory cache (CUDA) without unloading models."""
            self._update_heartbeat(request.client_id)

            with self._models_lock:
                if self._embedding_model:
                    if hasattr(self._embedding_model, "clear_cache"):
                        self._embedding_model.clear_cache()

                # Also clear reranker cache if loaded
                if self._reranker:
                    if hasattr(self._reranker, "clear_cache"):
                        self._reranker.clear_cache()

            return {"status": "ok"}

        @app.post("/rerank")
        async def rerank(request: RerankRequest):
            """Rerank documents."""
            self._update_heartbeat(request.client_id)

            # Use requested model or default from config
            target_model = request.model or self.config.get_reranker_model()

            # Check if reranking is enabled at all
            if not self.config.is_reranker_enabled() and not request.model:
                return {"error": "Reranker not enabled", "results": []}

            # Handle model loading/switching
            with self._models_lock:
                # Always ensure models are initialized (loads embedding + default reranker if none)
                self._load_models()

                # Switch reranker if different model requested
                if self._reranker is None or (
                    hasattr(self._reranker, "model_name")
                    and self._reranker.model_name != target_model
                ):
                    print(
                        f"Loading/Switching reranker to {target_model}...",
                        file=sys.stderr,
                    )
                    if self._reranker and hasattr(self._reranker, "unload_model"):
                        self._reranker.unload_model()

                    from .reranker.cross_encoder_reranker import CrossEncoderReranker

                    idle_timeout = self.config.get_model_idle_timeout()
                    self._reranker = CrossEncoderReranker(
                        target_model,
                        lazy_load=False,
                        idle_timeout=idle_timeout,
                    )

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._reranker.rerank(
                    request.query,
                    request.documents,
                    metadatas=request.metadatas,
                    top_k=request.top_k,
                ),
            )
            return {"results": results}

        @app.get("/config")
        async def get_config():
            """Get server configuration."""
            return {
                "embedding_model": self.config.get_embedding_model(),
                "reranker_enabled": self.config.is_reranker_enabled(),
                "reranker_model": (
                    self.config.get_reranker_model()
                    if self.config.is_reranker_enabled()
                    else None
                ),
            }

        return app

    def _update_heartbeat(self, client_id: str):
        """Update client heartbeat timestamp."""
        with self.clients_lock:
            self.clients[client_id] = time.time()

    async def _config_watcher(self):
        """Background task to watch for config file changes and reload."""
        while True:
            await asyncio.sleep(5)  # Check every 5 seconds
            try:
                if self.config.has_changed():
                    print("Config file change detected, reloading...", file=sys.stderr)
                    self.config.reload()
                    self._handle_config_reload()
            except Exception as e:
                print(f"Error checking for config changes: {e}", file=sys.stderr)

    def _handle_config_reload(self):
        """Handle necessary updates after config reload."""
        with self._models_lock:
            # Check if embedding model changed
            new_embedding_model = self.config.get_embedding_model()
            if (
                self._embedding_model
                and self._embedding_model.model_name != new_embedding_model
            ):
                print(
                    f"Embedding model changed to {new_embedding_model}, unloading old model...",
                    file=sys.stderr,
                )
                if hasattr(self._embedding_model, "unload_model"):
                    self._embedding_model.unload_model()
                self._embedding_model = None
                self._models_loaded = False

            # Check if reranker changed or was enabled/disabled
            new_reranker_model = self.config.get_reranker_model()
            reranker_enabled = self.config.is_reranker_enabled()

            if not reranker_enabled:
                if self._reranker:
                    print("Reranker disabled by config, unloading...", file=sys.stderr)
                    if hasattr(self._reranker, "unload_model"):
                        self._reranker.unload_model()
                    self._reranker = None
                    # No need to reset _models_loaded if embedding model is still fine
            else:
                if self._reranker and self._reranker.model_name != new_reranker_model:
                    print(
                        f"Reranker model changed to {new_reranker_model}, unloading old model...",
                        file=sys.stderr,
                    )
                    if hasattr(self._reranker, "unload_model"):
                        self._reranker.unload_model()
                    self._reranker = None
                    self._models_loaded = False
                elif not self._reranker:
                    # Reranker was enabled but not loaded yet
                    self._models_loaded = False

    async def _idle_checker(self):
        """Background task to check for idle state and shutdown."""
        while True:
            await asyncio.sleep(CHECK_INTERVAL)

            now = time.time()

            # Remove stale clients
            with self.clients_lock:
                stale_clients = [
                    cid
                    for cid, last_seen in self.clients.items()
                    if now - last_seen > HEARTBEAT_TIMEOUT
                ]
                for cid in stale_clients:
                    print(f"Client {cid} timed out", file=sys.stderr)
                    del self.clients[cid]

                active_clients = len(self.clients)

            # If no active clients, start shutdown countdown
            if active_clients == 0:
                print(
                    f"No active clients, waiting {IDLE_SHUTDOWN_DELAY}s before shutdown...",
                    file=sys.stderr,
                )
                await asyncio.sleep(IDLE_SHUTDOWN_DELAY)

                # Check again after delay
                with self.clients_lock:
                    if len(self.clients) == 0:
                        print(
                            "No clients reconnected, shutting down...", file=sys.stderr
                        )
                        self._cleanup()
                        os._exit(0)

    def _cleanup(self):
        """Clean up lock and info files."""
        try:
            get_lock_file_path().unlink(missing_ok=True)
            get_server_info_path().unlink(missing_ok=True)
        except Exception:
            pass

    def run(self):
        """Run the server."""
        _ensure_imports()

        # Write lock file
        lock_path = get_lock_file_path()
        info_path = get_server_info_path()

        lock_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if another server is running
        if lock_path.exists():
            try:
                with open(lock_path) as f:
                    pid = int(f.read().strip())
                # Check if process is alive
                os.kill(pid, 0)
                print(f"Another server is already running (PID {pid})", file=sys.stderr)
                sys.exit(1)
            except (ProcessLookupError, ValueError, FileNotFoundError):
                # Process doesn't exist, remove stale lock
                lock_path.unlink(missing_ok=True)

        # Find a free port (in case configured port is in use)
        try:
            actual_port = find_free_port(self.port)
            if actual_port != self.port:
                print(
                    f"Port {self.port} is in use, using port {actual_port}",
                    file=sys.stderr,
                )
            self.port = actual_port
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Write our PID to lock file
        with open(lock_path, "w") as f:
            f.write(str(os.getpid()))

        # Write server info with the actual port being used
        with open(info_path, "w") as f:
            json.dump({"port": self.port, "pid": os.getpid()}, f)

        # Handle signals for cleanup
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down...", file=sys.stderr)
            self._cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            app = self.create_app()
            print(f"Starting embedding server on port {self.port}", file=sys.stderr)
            uvicorn.run(app, host="127.0.0.1", port=self.port, log_level="warning")
        finally:
            self._cleanup()


def main():
    """Entry point for the embedding server."""
    import argparse

    parser = argparse.ArgumentParser(description="Code-RAG Embedding Server")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="Port to listen on"
    )
    args = parser.parse_args()

    server = EmbeddingServer(port=args.port)
    server.run()


if __name__ == "__main__":
    main()
