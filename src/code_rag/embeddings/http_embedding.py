"""HTTP client implementation of the embedding interface.

This client connects to the shared embedding server, spawning it on-demand
if not already running. Multiple MCP instances share the same model via HTTP.
"""

import atexit
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import requests

from .embedding_interface import EmbeddingInterface

# Server configuration
DEFAULT_PORT = 8199
STARTUP_TIMEOUT = 60  # Max time to wait for server to start (includes model loading)
HEARTBEAT_INTERVAL = 30  # Send heartbeat every 30 seconds
CONNECTION_TIMEOUT = 5  # HTTP request timeout


def get_server_info_path() -> Path:
    """Get the path to the server info file."""
    from ..config.config import Config

    config = Config()
    cache_dir = Path(config.get_database_path())
    return cache_dir / "embedding_server.json"


def get_server_log_path() -> Path:
    """Get the path to the server log file."""
    from ..config.config import Config

    config = Config()
    cache_dir = Path(config.get_database_path())
    return cache_dir / "embedding_server.log"


class HttpEmbedding(EmbeddingInterface):
    """HTTP client that connects to the shared embedding server."""

    def __init__(self, port: int = DEFAULT_PORT):
        """
        Initialize the HTTP embedding client.

        Args:
            port: Port where embedding server runs
        """
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.client_id = str(uuid.uuid4())

        self._heartbeat_thread: Optional[threading.Thread] = None
        self._heartbeat_stop = threading.Event()
        self._server_process: Optional[subprocess.Popen] = None
        self._dimension_cache: Optional[int] = None

        # Ensure server is running
        self._ensure_server()

        # Start heartbeat thread
        self._start_heartbeat()

        # Register cleanup on exit
        atexit.register(self._cleanup)

    def _is_server_running(self) -> bool:
        """Check if the embedding server is running and healthy."""
        try:
            response = requests.get(
                f"{self.base_url}/health", timeout=CONNECTION_TIMEOUT
            )
            return response.status_code == 200
        except (requests.ConnectionError, requests.Timeout):
            return False

    def _spawn_server(self):
        """Spawn the embedding server as a detached subprocess."""
        # Find the embedding_server module
        server_script = Path(__file__).parent.parent / "embedding_server.py"

        if not server_script.exists():
            raise RuntimeError(f"Embedding server script not found: {server_script}")

        # Spawn server as detached process
        # Use the same Python interpreter
        python_exe = sys.executable

        # Redirect stderr to a log file to avoid pipe buffer filling up and blocking
        log_path = get_server_log_path()

        # Rotate log if too large (10MB)
        try:
            if log_path.exists() and log_path.stat().st_size > 10 * 1024 * 1024:
                old_log = log_path.with_suffix(".log.old")
                log_path.rename(old_log)
        except Exception as e:
            print(f"Warning: Failed to rotate log file: {e}", file=sys.stderr)

        try:
            # Open in append mode
            log_file = open(log_path, "a")
        except Exception as e:
            print(f"Warning: Could not open log file {log_path}: {e}", file=sys.stderr)
            log_file = subprocess.DEVNULL

        try:
            # Create a detached subprocess that survives parent exit
            if os.name == "nt":  # Windows
                creationflags = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
                )
                self._server_process = subprocess.Popen(
                    [
                        python_exe,
                        "-m",
                        "code_rag.embedding_server",
                        "--port",
                        str(self.port),
                    ],
                    creationflags=creationflags,
                    stdout=subprocess.DEVNULL,
                    stderr=log_file,
                    cwd=str(Path(__file__).parent.parent.parent),
                )
            else:  # Unix
                # Use nohup-style detachment
                self._server_process = subprocess.Popen(
                    [
                        python_exe,
                        "-m",
                        "code_rag.embedding_server",
                        "--port",
                        str(self.port),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=log_file,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,  # Detach from parent
                    cwd=str(Path(__file__).parent.parent.parent),
                )
        finally:
            if log_file != subprocess.DEVNULL:
                log_file.close()

        print(
            f"Spawned embedding server (PID {self._server_process.pid}) logging to {log_path}",
            file=sys.stderr,
        )

    def _ensure_server(self):
        """Ensure the embedding server is running, spawning if needed.

        The server may choose a different port than configured if the default is busy,
        so we always read the actual port from the server info file.
        """
        info_path = get_server_info_path()

        # First, check if server info file exists and try connecting to that port
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                actual_port = info.get("port", self.port)
                self.port = actual_port
                self.base_url = f"http://127.0.0.1:{self.port}"

                if self._is_server_running():
                    return
            except (json.JSONDecodeError, FileNotFoundError):
                pass

        # Server not running, spawn it
        print("Embedding server not running, spawning...", file=sys.stderr)
        self._spawn_server()

        # Wait for server to become healthy and write its info file
        start_time = time.time()
        while time.time() - start_time < STARTUP_TIMEOUT:
            # Re-read the info file to get the actual port the server chose
            if info_path.exists():
                try:
                    with open(info_path) as f:
                        info = json.load(f)
                    actual_port = info.get("port", self.port)
                    if actual_port != self.port:
                        print(f"Server is using port {actual_port}", file=sys.stderr)
                        self.port = actual_port
                        self.base_url = f"http://127.0.0.1:{self.port}"
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            if self._is_server_running():
                print("Embedding server is ready", file=sys.stderr)
                return

            time.sleep(0.5)

        # Check if process died
        if self._server_process and self._server_process.poll() is not None:
            # Read last few lines of log file for error details
            stderr_output = "Check log file for details"
            try:
                log_path = get_server_log_path()
                if log_path.exists():
                    # Read last 1KB
                    size = log_path.stat().st_size
                    with open(log_path, "r") as f:
                        if size > 1024:
                            f.seek(size - 1024)
                        stderr_output = f.read()
            except Exception:
                pass

            raise RuntimeError(
                f"Embedding server failed to start. Exit code: {self._server_process.returncode}\n"
                f"Log tail: {stderr_output}"
            )

        raise RuntimeError(
            f"Embedding server failed to start within {STARTUP_TIMEOUT}s"
        )

    def _start_heartbeat(self):
        """Start the heartbeat thread."""
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True, name="embedding-heartbeat"
        )
        self._heartbeat_thread.start()

    def _heartbeat_loop(self):
        """Send periodic heartbeats to the server."""
        while not self._heartbeat_stop.is_set():
            try:
                requests.post(
                    f"{self.base_url}/heartbeat",
                    json={"client_id": self.client_id},
                    timeout=CONNECTION_TIMEOUT,
                )
            except (requests.ConnectionError, requests.Timeout):
                # Server might have died, try to restart on next operation
                pass

            self._heartbeat_stop.wait(HEARTBEAT_INTERVAL)

    def _cleanup(self):
        """Clean up on exit."""
        # Stop heartbeat thread
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1)

        # Notify server we're disconnecting
        try:
            requests.post(
                f"{self.base_url}/disconnect",
                json={"client_id": self.client_id},
                timeout=2,
            )
        except (requests.ConnectionError, requests.Timeout):
            pass

    def _request(self, endpoint: str, data: dict, retry: bool = True) -> dict:
        """Make a request to the server with automatic reconnection."""
        data["client_id"] = self.client_id

        try:
            response = requests.post(
                f"{self.base_url}/{endpoint}",
                json=data,
                timeout=120,  # Long timeout for embedding operations
            )
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout) as e:
            if retry:
                # Server might have died, try to restart
                self._ensure_server()
                return self._request(endpoint, data, retry=False)
            raise RuntimeError(f"Failed to connect to embedding server: {e}")

    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        result = self._request("embed", {"texts": [text]})
        return result["embeddings"][0]

    def embed_query(self, query: str) -> List[float]:
        """Embed a query with model-specific prefix."""
        result = self._request("embed_query", {"query": query})
        return result["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        result = self._request("embed", {"texts": texts})
        return result["embeddings"]

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        if self._dimension_cache is not None:
            return self._dimension_cache

        result = self._request("embedding_dimension", {})
        self._dimension_cache = result["dimension"]
        return self._dimension_cache

    def start_background_loading(self):
        """No-op for HTTP client - server handles model loading."""
        pass

    def unload_model(self):
        """No-op for HTTP client - server manages model lifecycle."""
        pass

    def clear_cache(self):
        """Request server to clear memory cache."""
        try:
            # We don't retry this if it fails, as it's an optimization only
            requests.post(
                f"{self.base_url}/clear_cache",
                json={"client_id": self.client_id},
                timeout=CONNECTION_TIMEOUT,
            )
        except (requests.ConnectionError, requests.Timeout):
            pass
