"""HTTP client implementation of the reranker interface.

This client connects to the shared embedding server for reranking,
sharing the model with other MCP instances.
"""

from typing import List, Tuple

import requests

from .reranker_interface import RerankerInterface

# Use same port as embedding server
DEFAULT_PORT = 8199
CONNECTION_TIMEOUT = 120  # Long timeout for reranking operations


class HttpReranker(RerankerInterface):
    """HTTP client that connects to the shared embedding server for reranking."""

    def __init__(self, port: int = DEFAULT_PORT, client_id: str = ""):
        """
        Initialize the HTTP reranker client.

        Args:
            port: Port where embedding server runs
            client_id: Client ID for heartbeat tracking (should match HttpEmbedding client_id)
        """
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.client_id = client_id

    def rerank(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using the shared server.

        Args:
            query: The search query string
            documents: List of document texts to rerank
            top_k: Number of top results to return

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        if not documents:
            return []

        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                json={
                    "query": query,
                    "documents": documents,
                    "top_k": top_k,
                    "client_id": self.client_id,
                },
                timeout=CONNECTION_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()

            if "error" in result:
                return []

            # Convert list of lists back to list of tuples
            return [(idx, score) for idx, score in result["results"]]

        except (requests.ConnectionError, requests.Timeout):
            # If server is down, return empty (caller should handle gracefully)
            return []

    def start_background_loading(self):
        """No-op for HTTP client - server handles model loading."""
        pass

    def unload_model(self):
        """No-op for HTTP client - server manages model lifecycle."""
        pass
