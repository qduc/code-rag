from unittest.mock import MagicMock, patch

import pytest

from code_rag.mcp_server import call_tool


@pytest.mark.asyncio
async def test_mcp_search_codebase_reranking_switch():
    """Test that the search_codebase tool correctly passes the rerank parameter."""
    mock_api = MagicMock()
    # Mock search results
    mock_api.search.return_value = [
        {
            "content": "test content",
            "file_path": "test.py",
            "similarity": 0.9,
            "chunk_index": 0,
            "total_chunks": 1,
            "start_line": 1,
            "end_line": 2,
        }
    ]
    # Mock ensure_indexed
    mock_api.ensure_indexed.return_value = {"success": True}

    with (
        patch("code_rag.mcp_server.api", mock_api),
        patch("code_rag.mcp_server.api_ready_event.wait", return_value=True),
    ):

        # 1. Test with reranking enabled (explicitly)
        arguments = {
            "codebase_path": "/test/path",
            "query": "test query",
            "enable_reranking": True,
        }
        await call_tool("search_codebase", arguments)

        # Check that api.search was called with rerank=True
        mock_api.search.assert_called_with(
            "test query",
            n_results=5,
            expand_context=False,
            file_types=None,
            include_paths=None,
            rerank=True,
            reranker_multiplier=2,
            reranker_model=None,
        )

        # 2. Test with reranking disabled (explicitly)
        mock_api.search.reset_mock()
        arguments = {
            "codebase_path": "/test/path",
            "query": "test query",
            "enable_reranking": False,
        }
        await call_tool("search_codebase", arguments)

        # Check that api.search was called with rerank=False
        mock_api.search.assert_called_with(
            "test query",
            n_results=5,
            expand_context=False,
            file_types=None,
            include_paths=None,
            rerank=False,
            reranker_multiplier=2,
            reranker_model=None,
        )

        # 3. Test default (should be False)
        mock_api.search.reset_mock()
        arguments = {"codebase_path": "/test/path", "query": "test query"}
        await call_tool("search_codebase", arguments)

        # Check that api.search was called with rerank=False (from arguments.get default)
        mock_api.search.assert_called_with(
            "test query",
            n_results=5,
            expand_context=False,
            file_types=None,
            include_paths=None,
            rerank=False,
            reranker_multiplier=2,
            reranker_model=None,
        )

        # 4. Test with custom reranker_multiplier
        mock_api.search.reset_mock()
        arguments = {
            "codebase_path": "/test/path",
            "query": "test query",
            "enable_reranking": True,
            "reranker_multiplier": 5,
        }
        await call_tool("search_codebase", arguments)

        # Check that api.search was called with reranker_multiplier=5
        mock_api.search.assert_called_with(
            "test query",
            n_results=5,
            expand_context=False,
            file_types=None,
            include_paths=None,
            rerank=True,
            reranker_multiplier=5,
            reranker_model=None,
        )
        # 5. Test with custom reranker_model
        mock_api.search.reset_mock()
        arguments = {
            "codebase_path": "/test/path",
            "query": "test query",
            "enable_reranking": True,
            "reranker_model": "custom-reranker",
        }
        await call_tool("search_codebase", arguments)

        # Check that api.search was called with reranker_model="custom-reranker"
        mock_api.search.assert_called_with(
            "test query",
            n_results=5,
            expand_context=False,
            file_types=None,
            include_paths=None,
            rerank=True,
            reranker_multiplier=2,
            reranker_model="custom-reranker",
        )


@pytest.mark.asyncio
async def test_api_search_rerank_parameter():
    """Test that CodeRAGAPI.search respects the rerank parameter."""
    from code_rag.api import CodeRAGAPI

    with (
        patch("code_rag.api.Config") as MockConfig,
        patch("code_rag.api.OpenAIEmbedding"),
        patch("code_rag.api.ChromaDatabase"),
        patch("code_rag.api.CrossEncoderReranker") as MockRerankerClass,
    ):
        # Setup mock config
        mock_config_instance = MockConfig.return_value
        mock_config_instance.is_shared_server_enabled.return_value = False
        mock_config_instance.get_database_path.return_value = "/tmp/test"
        mock_config_instance.get_model_idle_timeout.return_value = 1800

        # Setup mock reranker instance
        mock_reranker = MockRerankerClass.return_value
        mock_reranker.rerank.return_value = [(0, 0.95)]
        mock_reranker.model_name = "default"

        api = CodeRAGAPI(reranker_enabled=True)
        # Verify initial reranker
        assert api.reranker == mock_reranker

        # Mock database query
        api.database.query.return_value = {
            "documents": [["doc1"]],
            "metadatas": [[{"file_path": "f1.py"}]],
            "distances": [[0.1]],
        }
        api._active_collection = "test"

        # 1. Call search with rerank=True (default model)
        api.search("query", rerank=True)
        assert mock_reranker.rerank.called

        # 2. Call search with rerank=False
        mock_reranker.rerank.reset_mock()
        api.search("query", rerank=False)
        assert not mock_reranker.rerank.called

        # 3. Call search with custom model
        # Create a new mock for the new reranker instance that will be created
        new_mock_reranker = MagicMock()
        new_mock_reranker.rerank.return_value = [(0, 0.85)]
        new_mock_reranker.model_name = "custom-model"
        MockRerankerClass.return_value = new_mock_reranker

        api.search("query", rerank=True, reranker_model="custom-model")

        # Verify it used the NEW reranker
        assert api.reranker == new_mock_reranker
        new_mock_reranker.rerank.assert_called()
        _, kwargs = new_mock_reranker.rerank.call_args
        assert kwargs.get("model") == "custom-model"
