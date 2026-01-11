"""Unit tests for CodeRAGAPI initialization and factory methods.

These tests verify:
1. CodeRAGAPI __init__ - different configuration options
2. _create_embedding_model - factory for embedding backends
3. _generate_collection_name (module-level) - collection name generation logic

All tests mock external dependencies to avoid loading real models.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from code_rag.api import CodeRAGAPI, generate_collection_name

# ============================================================================
# generate_collection_name Tests
# ============================================================================


class TestGenerateCollectionName:
    """Tests for the generate_collection_name function."""

    def test_generates_consistent_name_for_same_path(self):
        """Test that same path always generates the same collection name."""
        path = "/home/user/myproject"
        name1 = generate_collection_name(path)
        name2 = generate_collection_name(path)
        assert name1 == name2

    def test_different_paths_generate_different_names(self):
        """Test that different paths generate different collection names."""
        name1 = generate_collection_name("/home/user/project1")
        name2 = generate_collection_name("/home/user/project2")
        assert name1 != name2

    def test_name_starts_with_codebase_prefix(self):
        """Test that generated names have the expected prefix."""
        name = generate_collection_name("/some/path")
        assert name.startswith("codebase_")

    def test_name_has_consistent_length(self):
        """Test that generated names have consistent length (prefix + 16 char hash)."""
        name = generate_collection_name("/any/path")
        # "codebase_" is 9 chars + 16 char hash = 25 chars
        assert len(name) == 25

    def test_resolves_relative_paths(self):
        """Test that relative paths are resolved to absolute paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a subdirectory
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()

            # Get absolute path name
            abs_name = generate_collection_name(str(subdir))

            # Test with different relative path representations
            # would resolve to same absolute path
            assert abs_name.startswith("codebase_")

    def test_handles_trailing_slashes(self):
        """Test that trailing slashes don't affect the result after resolve."""
        # Note: Path.resolve() normalizes trailing slashes
        path1 = "/home/user/project"
        path2 = "/home/user/project/"

        name1 = generate_collection_name(path1)
        name2 = generate_collection_name(path2)

        # Both should resolve to same path
        assert name1 == name2


# ============================================================================
# CodeRAGAPI.__init__ Tests
# ============================================================================


class TestCodeRAGAPIInit:
    """Tests for CodeRAGAPI initialization."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies for CodeRAGAPI init."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.QdrantDatabase") as mock_qdrant,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            # Setup mock config
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock embedding model
            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_st.return_value = mock_embedding

            # Setup mock databases
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            mock_qdrant_instance = MagicMock()
            mock_qdrant.return_value = mock_qdrant_instance

            # Setup mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            yield {
                "SentenceTransformerEmbedding": mock_st,
                "ChromaDatabase": mock_chroma,
                "QdrantDatabase": mock_qdrant,
                "CrossEncoderReranker": mock_reranker,
                "Config": mock_config,
                "config_instance": mock_config_instance,
                "embedding_instance": mock_embedding,
                "chroma_instance": mock_chroma_instance,
                "qdrant_instance": mock_qdrant_instance,
                "reranker_instance": mock_reranker_instance,
            }

    def test_default_initialization(self, mock_dependencies):
        """Test CodeRAGAPI initializes with default parameters."""
        api = CodeRAGAPI()

        assert api.database_type == "chroma"
        assert api.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert api.reranker_enabled is True
        assert api.reranker_multiplier == 2
        assert api.lazy_load_models is False

    def test_custom_database_type_chroma(self, mock_dependencies):
        """Test initialization with chroma database."""
        api = CodeRAGAPI(database_type="chroma")

        mock_dependencies["ChromaDatabase"].assert_called_once()
        mock_dependencies["QdrantDatabase"].assert_not_called()
        assert api.database_type == "chroma"

    def test_custom_database_type_qdrant(self, mock_dependencies):
        """Test initialization with qdrant database."""
        api = CodeRAGAPI(database_type="qdrant")

        mock_dependencies["QdrantDatabase"].assert_called_once()
        assert api.database_type == "qdrant"

    def test_custom_database_path(self, mock_dependencies):
        """Test initialization with custom database path."""
        custom_path = "/custom/db/path"
        api = CodeRAGAPI(database_path=custom_path)

        assert api.database_path == custom_path
        mock_dependencies["ChromaDatabase"].assert_called_with(
            persist_directory=custom_path
        )

    def test_default_database_path_from_config(self, mock_dependencies):
        """Test database path falls back to config default."""
        api = CodeRAGAPI()

        assert api.database_path == "/tmp/test-db"
        mock_dependencies["config_instance"].get_database_path.assert_called()

    def test_custom_embedding_model(self, mock_dependencies):
        """Test initialization with custom embedding model."""
        custom_model = "custom/embedding-model"
        api = CodeRAGAPI(embedding_model=custom_model)

        assert api.embedding_model_name == custom_model
        mock_dependencies["SentenceTransformerEmbedding"].assert_called_with(
            custom_model, lazy_load=False, idle_timeout=1800
        )

    def test_reranker_disabled(self, mock_dependencies):
        """Test initialization with reranker disabled."""
        api = CodeRAGAPI(reranker_enabled=False)

        assert api.reranker_enabled is False
        assert api.reranker is None
        mock_dependencies["CrossEncoderReranker"].assert_not_called()

    def test_reranker_enabled_default(self, mock_dependencies):
        """Test reranker is enabled by default."""
        api = CodeRAGAPI()

        assert api.reranker_enabled is True
        mock_dependencies["CrossEncoderReranker"].assert_called_once()

    def test_custom_reranker_model(self, mock_dependencies):
        """Test initialization with custom reranker model."""
        custom_model = "custom/reranker-model"
        api = CodeRAGAPI(reranker_model=custom_model)

        mock_dependencies["CrossEncoderReranker"].assert_called_with(
            custom_model, lazy_load=False, idle_timeout=1800
        )

    def test_reranker_multiplier(self, mock_dependencies):
        """Test initialization with custom reranker multiplier."""
        api = CodeRAGAPI(reranker_multiplier=5)

        assert api.reranker_multiplier == 5

    def test_lazy_load_models_enabled(self, mock_dependencies):
        """Test initialization with lazy loading enabled."""
        api = CodeRAGAPI(lazy_load_models=True)

        assert api.lazy_load_models is True
        mock_dependencies["SentenceTransformerEmbedding"].assert_called_with(
            "sentence-transformers/all-MiniLM-L6-v2", lazy_load=True, idle_timeout=1800
        )

    def test_session_state_initialized(self, mock_dependencies):
        """Test that session state is properly initialized."""
        api = CodeRAGAPI()

        assert api._indexed_paths == set()
        assert api._active_collection is None
        assert api._metadata_indices == {}


class TestCodeRAGAPIInitSharedServer:
    """Tests for CodeRAGAPI initialization with shared server mode."""

    def test_uses_http_embedding_when_shared_server_enabled(self):
        """Test that HTTP embedding is used when shared server is enabled."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch(
                "code_rag.embeddings.http_embedding.HttpEmbedding"
            ) as mock_http_embed,
        ):

            # Setup mock config for shared server mode
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = True
            mock_config_instance.get_shared_server_port.return_value = 9999
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock HTTP embedding
            mock_http_embed_instance = MagicMock()
            mock_http_embed_instance.get_embedding_dimension.return_value = 384
            mock_http_embed_instance.client_id = "test-client-id"
            mock_http_embed.return_value = mock_http_embed_instance

            # Setup mock database
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            api = CodeRAGAPI(reranker_enabled=False)

            assert api._use_shared_server is True
            # SentenceTransformerEmbedding should NOT be called in shared server mode
            mock_st.assert_not_called()
            # HttpEmbedding should be called with the port
            mock_http_embed.assert_called_with(port=9999)

    def test_shared_server_port_from_config(self):
        """Test that shared server port is read from config."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch(
                "code_rag.embeddings.http_embedding.HttpEmbedding"
            ) as mock_http_embed,
        ):

            # Setup mock config for shared server mode
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = True
            mock_config_instance.get_shared_server_port.return_value = 7777
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock HTTP embedding
            mock_http_embed_instance = MagicMock()
            mock_http_embed_instance.get_embedding_dimension.return_value = 384
            mock_http_embed.return_value = mock_http_embed_instance

            # Setup mock database
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            api = CodeRAGAPI(reranker_enabled=False)

            assert api._shared_server_port == 7777


# ============================================================================
# CodeRAGAPI._create_embedding_model Tests
# ============================================================================


class TestCreateEmbeddingModel:
    """Tests for CodeRAGAPI._create_embedding_model factory method."""

    @pytest.fixture
    def api_with_mocks(self):
        """Create a CodeRAGAPI instance with mocked components for testing _create_embedding_model."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.LiteLLMEmbedding") as mock_litellm,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            # Setup mock config
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock embedding model
            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_st.return_value = mock_embedding
            mock_litellm.return_value = mock_embedding

            # Setup mock database
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            # Setup mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            api = CodeRAGAPI(reranker_enabled=False)

            yield {
                "api": api,
                "SentenceTransformerEmbedding": mock_st,
                "LiteLLMEmbedding": mock_litellm,
                "config_instance": mock_config_instance,
            }

    def test_creates_sentence_transformer_for_local_model(self, api_with_mocks):
        """Test that SentenceTransformerEmbedding is created for local models."""
        api = api_with_mocks["api"]
        mock_st = api_with_mocks["SentenceTransformerEmbedding"]

        # Reset mock to clear initialization call
        mock_st.reset_mock()

        api._create_embedding_model("sentence-transformers/all-MiniLM-L6-v2")

        mock_st.assert_called_once_with(
            "sentence-transformers/all-MiniLM-L6-v2", lazy_load=False, idle_timeout=1800
        )

    def test_creates_litellm_for_openai_text_embedding(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for OpenAI text-embedding models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("text-embedding-3-small")

        mock_litellm.assert_called_once_with(
            "text-embedding-3-small", idle_timeout=1800
        )

    def test_creates_litellm_for_openai_prefix(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for openai/ prefixed models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("openai/text-embedding-ada-002")

        mock_litellm.assert_called_once_with(
            "openai/text-embedding-ada-002", idle_timeout=1800
        )

    def test_creates_litellm_for_azure_prefix(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for azure/ prefixed models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("azure/text-embedding-3-small")

        mock_litellm.assert_called_once_with(
            "azure/text-embedding-3-small", idle_timeout=1800
        )

    def test_creates_litellm_for_vertex_ai_prefix(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for vertex_ai/ prefixed models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("vertex_ai/text-embedding-004")

        mock_litellm.assert_called_once_with(
            "vertex_ai/text-embedding-004", idle_timeout=1800
        )

    def test_creates_litellm_for_cohere_prefix(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for cohere/ prefixed models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("cohere/embed-english-v3.0")

        mock_litellm.assert_called_once_with(
            "cohere/embed-english-v3.0", idle_timeout=1800
        )

    def test_creates_litellm_for_bedrock_prefix(self, api_with_mocks):
        """Test that LiteLLMEmbedding is created for bedrock/ prefixed models."""
        api = api_with_mocks["api"]
        mock_litellm = api_with_mocks["LiteLLMEmbedding"]

        mock_litellm.reset_mock()

        api._create_embedding_model("bedrock/amazon.titan-embed-text-v1")

        mock_litellm.assert_called_once_with(
            "bedrock/amazon.titan-embed-text-v1", idle_timeout=1800
        )

    def test_creates_sentence_transformer_for_unknown_prefix(self, api_with_mocks):
        """Test that SentenceTransformerEmbedding is created for unknown prefixes."""
        api = api_with_mocks["api"]
        mock_st = api_with_mocks["SentenceTransformerEmbedding"]

        mock_st.reset_mock()

        api._create_embedding_model("nomic-ai/CodeRankEmbed")

        mock_st.assert_called_once_with(
            "nomic-ai/CodeRankEmbed", lazy_load=False, idle_timeout=1800
        )

    def test_lazy_load_passed_to_sentence_transformer(self, api_with_mocks):
        """Test that lazy_load parameter is correctly passed."""
        api = api_with_mocks["api"]
        mock_st = api_with_mocks["SentenceTransformerEmbedding"]

        mock_st.reset_mock()

        api._create_embedding_model("some-local-model", lazy_load=True)

        mock_st.assert_called_once_with(
            "some-local-model", lazy_load=True, idle_timeout=1800
        )


class TestCreateEmbeddingModelSharedServer:
    """Tests for _create_embedding_model with shared server mode."""

    def test_creates_http_embedding_when_shared_server_enabled(self):
        """Test that HttpEmbedding is created when shared server is enabled."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch(
                "code_rag.embeddings.http_embedding.HttpEmbedding"
            ) as mock_http_embed,
        ):

            # Setup mock config for shared server mode
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = True
            mock_config_instance.get_shared_server_port.return_value = 9999
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock HTTP embedding
            mock_http_embed_instance = MagicMock()
            mock_http_embed_instance.get_embedding_dimension.return_value = 384
            mock_http_embed.return_value = mock_http_embed_instance

            # Setup mock database
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            api = CodeRAGAPI(reranker_enabled=False)

            # HttpEmbedding should have been called during init
            mock_http_embed.assert_called_with(port=9999)
            # SentenceTransformerEmbedding should NOT have been called
            mock_st.assert_not_called()


# ============================================================================
# CodeRAGAPI._create_database Tests
# ============================================================================


class TestCreateDatabase:
    """Tests for CodeRAGAPI._create_database factory method."""

    @pytest.fixture
    def api_instance(self):
        """Create a minimal CodeRAGAPI instance for testing _create_database."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_st.return_value = mock_embedding

            mock_chroma.return_value = MagicMock()

            api = CodeRAGAPI(reranker_enabled=False)

            yield api

    def test_creates_chroma_database(self, api_instance):
        """Test that ChromaDatabase is created for 'chroma' type."""
        with patch("code_rag.api.ChromaDatabase") as mock_chroma:
            mock_chroma.return_value = MagicMock()

            result = api_instance._create_database("chroma", "/test/path")

            mock_chroma.assert_called_with(persist_directory="/test/path")

    def test_creates_qdrant_database(self, api_instance):
        """Test that QdrantDatabase is created for 'qdrant' type."""
        with patch("code_rag.api.QdrantDatabase") as mock_qdrant:
            mock_qdrant.return_value = MagicMock()

            result = api_instance._create_database("qdrant", "/test/path")

            mock_qdrant.assert_called_with(persist_directory="/test/path")

    def test_raises_for_unsupported_database_type(self, api_instance):
        """Test that ValueError is raised for unsupported database types."""
        with pytest.raises(ValueError, match="Unsupported database type: invalid"):
            api_instance._create_database("invalid", "/test/path")


# ============================================================================
# CodeRAGAPI Error Handling Tests
# ============================================================================


class TestCodeRAGAPIErrorHandling:
    """Tests for error handling during CodeRAGAPI initialization."""

    def test_reranker_failure_disables_reranker(self):
        """Test that reranker initialization failure gracefully disables reranker."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_st.return_value = mock_embedding

            mock_chroma.return_value = MagicMock()

            # Make reranker initialization fail
            mock_reranker.side_effect = RuntimeError("Failed to load reranker model")

            # Should not raise - just disables reranker
            api = CodeRAGAPI(reranker_enabled=True)

            assert api.reranker is None


# ============================================================================
# CodeRAGAPI.search Tests
# ============================================================================


class TestCodeRAGAPISearch:
    """Tests for CodeRAGAPI.search method."""

    @pytest.fixture
    def mock_api(self):
        """Create a CodeRAGAPI instance with mocked components for search tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            # Setup mock config
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            # Setup mock embedding model
            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_embedding.embed_query.return_value = [0.1] * 384
            mock_embedding.embed.return_value = [0.1] * 384
            mock_st.return_value = mock_embedding

            # Setup mock database with sample results
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.query.return_value = {
                "documents": [["def hello():\n    print('world')"]],
                "metadatas": [
                    [
                        {
                            "file_path": "/test/hello.py",
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "start_line": 1,
                            "end_line": 2,
                            "function_name": "hello",
                            "symbol_type": "function",
                        }
                    ]
                ],
                "distances": [[0.1]],
            }
            mock_chroma.return_value = mock_chroma_instance

            # Setup mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker_instance.rerank.return_value = [(0, 0.95)]
            mock_reranker.return_value = mock_reranker_instance

            api = CodeRAGAPI()
            api._active_collection = "test_collection"

            yield {
                "api": api,
                "embedding": mock_embedding,
                "database": mock_chroma_instance,
                "reranker": mock_reranker_instance,
            }

    def test_search_requires_active_collection(self, mock_api):
        """Test that search raises error when no collection is active."""
        api = mock_api["api"]
        api._active_collection = None

        with pytest.raises(ValueError, match="No collection specified"):
            api.search("test query")

    def test_search_uses_specified_collection(self, mock_api):
        """Test that search uses specified collection over active collection."""
        api = mock_api["api"]
        api._active_collection = "default_collection"

        # Should not raise
        api.search("test query", collection_name="specific_collection")

    def test_search_generates_query_embedding(self, mock_api):
        """Test that search generates embedding for query."""
        api = mock_api["api"]
        mock_embedding = mock_api["embedding"]

        api.search("find authentication code")

        mock_embedding.embed_query.assert_called_with("find authentication code")

    def test_search_returns_formatted_results(self, mock_api):
        """Test that search returns properly formatted results."""
        api = mock_api["api"]

        results = api.search("hello function", rerank=False)

        assert len(results) == 1
        assert results[0]["content"] == "def hello():\n    print('world')"
        assert results[0]["file_path"] == "/test/hello.py"
        assert results[0]["chunk_index"] == 0
        assert results[0]["function_name"] == "hello"
        assert "similarity" in results[0]

    def test_search_with_reranking_enabled(self, mock_api):
        """Test search with reranking enabled."""
        api = mock_api["api"]
        mock_reranker = mock_api["reranker"]

        results = api.search("hello function", rerank=True)

        mock_reranker.rerank.assert_called_once()
        # Reranker score should be used as similarity
        assert results[0]["similarity"] == 0.95

    def test_search_with_reranking_disabled(self, mock_api):
        """Test search with reranking disabled."""
        api = mock_api["api"]
        mock_reranker = mock_api["reranker"]

        results = api.search("hello function", rerank=False)

        mock_reranker.rerank.assert_not_called()
        # Similarity should be calculated from distance
        assert results[0]["similarity"] == 0.9  # 1 - 0.1 distance

    def test_search_with_custom_n_results(self, mock_api):
        """Test search respects n_results parameter."""
        api = mock_api["api"]
        mock_db = mock_api["database"]

        api.search("test query", n_results=10, rerank=False)

        mock_db.query.assert_called()
        call_args = mock_db.query.call_args
        assert call_args[1]["n_results"] == 10

    def test_search_with_reranker_multiplier(self, mock_api):
        """Test search uses reranker multiplier for initial retrieval."""
        api = mock_api["api"]
        api.reranker_multiplier = 3
        mock_db = mock_api["database"]

        api.search("test query", n_results=5, rerank=True)

        mock_db.query.assert_called()
        call_args = mock_db.query.call_args
        # Should retrieve 5 * 3 = 15 results for reranking
        assert call_args[1]["n_results"] == 15

    def test_search_with_custom_reranker_multiplier(self, mock_api):
        """Test search uses custom reranker multiplier when provided."""
        api = mock_api["api"]
        api.reranker_multiplier = 2  # Default
        mock_db = mock_api["database"]

        api.search("test query", n_results=5, rerank=True, reranker_multiplier=4)

        mock_db.query.assert_called()
        call_args = mock_db.query.call_args
        # Should use custom multiplier: 5 * 4 = 20
        assert call_args[1]["n_results"] == 20

    def test_search_with_file_type_filter(self, mock_api):
        """Test search filters results by file type."""
        api = mock_api["api"]
        mock_db = mock_api["database"]

        # Setup multiple results with different file types
        mock_db.query.return_value = {
            "documents": [["py content", "js content", "md content"]],
            "metadatas": [
                [
                    {"file_path": "/test/file.py", "chunk_index": 0, "total_chunks": 1},
                    {"file_path": "/test/file.js", "chunk_index": 0, "total_chunks": 1},
                    {"file_path": "/test/file.md", "chunk_index": 0, "total_chunks": 1},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        results = api.search("test query", file_types=[".py"], rerank=False)

        # Should only return Python file
        assert len(results) == 1
        assert results[0]["file_path"] == "/test/file.py"

    def test_search_with_include_paths_filter(self, mock_api):
        """Test search filters results by path patterns."""
        api = mock_api["api"]
        mock_db = mock_api["database"]

        mock_db.query.return_value = {
            "documents": [["src content", "test content", "docs content"]],
            "metadatas": [
                [
                    {
                        "file_path": "/project/src/main.py",
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                    {
                        "file_path": "/project/tests/test_main.py",
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                    {
                        "file_path": "/project/docs/readme.md",
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }

        results = api.search("test query", include_paths=["src/"], rerank=False)

        # Should only return file from src/
        assert len(results) == 1
        assert "src/main.py" in results[0]["file_path"]

    def test_search_empty_results(self, mock_api):
        """Test search handles empty results gracefully."""
        api = mock_api["api"]
        mock_db = mock_api["database"]

        mock_db.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        results = api.search("nonexistent query")

        assert results == []

    def test_search_reranker_failure_fallback(self, mock_api):
        """Test search falls back to original results when reranker fails."""
        api = mock_api["api"]
        mock_reranker = mock_api["reranker"]

        # Make reranker fail
        mock_reranker.rerank.side_effect = RuntimeError("Reranker failed")

        # Should not raise, should return results from original database query
        results = api.search("test query", rerank=True)

        assert len(results) == 1
        # Results should still be returned (fallback to original distance)
        # Note: similarity calculation still uses reranker path since reranker is not None
        assert results[0]["content"] == "def hello():\n    print('world')"


class TestCodeRAGAPISearchIdentifierBoosting:
    """Tests for identifier-based boosting in search."""

    @pytest.fixture
    def mock_api_with_results(self):
        """Create API with multiple search results for boosting tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_embedding.embed_query.return_value = [0.1] * 384
            mock_st.return_value = mock_embedding

            # Multiple results - one containing the identifier, one not
            mock_chroma_instance = MagicMock()
            mock_chroma_instance.query.return_value = {
                "documents": [
                    [
                        "def process_data(): pass",  # Does not contain "getUserName"
                        "def getUserName(): return name",  # Contains the identifier
                    ]
                ],
                "metadatas": [
                    [
                        {
                            "file_path": "/test/a.py",
                            "chunk_index": 0,
                            "total_chunks": 1,
                        },
                        {
                            "file_path": "/test/b.py",
                            "chunk_index": 0,
                            "total_chunks": 1,
                        },
                    ]
                ],
                "distances": [[0.2, 0.3]],  # First result is closer in vector space
            }
            mock_chroma.return_value = mock_chroma_instance

            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            api = CodeRAGAPI(reranker_enabled=False)
            api._active_collection = "test_collection"

            yield {
                "api": api,
                "database": mock_chroma_instance,
            }

    def test_search_boosts_identifier_matches(self, mock_api_with_results):
        """Test that search boosts results containing query identifiers."""
        api = mock_api_with_results["api"]

        # Query contains a camelCase identifier
        # Use rerank=False since we're testing identifier boosting, not reranking
        results = api.search("getUserName function", n_results=2, rerank=False)

        # Result containing "getUserName" should be boosted
        boosted_result = next(r for r in results if "getUserName" in r["content"])
        non_boosted = next(r for r in results if "getUserName" not in r["content"])

        assert boosted_result["boosted"] is True
        assert non_boosted["boosted"] is False

    def test_search_marks_unboosted_results(self, mock_api_with_results):
        """Test that results without identifier matches are marked as not boosted."""
        api = mock_api_with_results["api"]

        # Query without identifiers
        results = api.search("find some code", n_results=2)

        # All results should not be boosted
        for result in results:
            assert result["boosted"] is False


# ============================================================================
# CodeRAGAPI.index_codebase Tests
# ============================================================================


class TestCodeRAGAPIIndexCodebase:
    """Tests for CodeRAGAPI.index_codebase method."""

    @pytest.fixture
    def mock_api_for_indexing(self):
        """Create a CodeRAGAPI instance with mocked components for indexing tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch("code_rag.api.FileProcessor") as mock_processor,
            patch("code_rag.api.MetadataIndex") as mock_metadata_index,
        ):

            # Setup mock config
            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config_instance.should_exclude_tests.return_value = False
            mock_config_instance.should_include_file_header.return_value = True
            mock_config_instance.get_additional_ignore_patterns.return_value = []
            mock_config_instance.get_chunk_size.return_value = 1500
            mock_config_instance.get_batch_size.return_value = 100
            mock_config_instance.should_verify_changes_with_hash.return_value = False
            mock_config.return_value = mock_config_instance

            # Setup mock embedding model
            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_embedding.embed_batch.return_value = [[0.1] * 384]
            mock_embedding.clear_cache.return_value = None
            mock_st.return_value = mock_embedding

            # Setup mock database
            mock_chroma_instance = MagicMock()
            mock_chroma.return_value = mock_chroma_instance

            # Setup mock reranker
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            # Setup mock file processor
            mock_processor_instance = MagicMock()
            mock_processor_instance.discover_files.return_value = ["/test/file1.py"]
            mock_processor_instance.process_file.return_value = [
                {
                    "id": "chunk_0",
                    "content": "def test(): pass",
                    "metadata": {
                        "file_path": "/test/file1.py",
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                }
            ]
            mock_processor_instance.get_file_stats.return_value = {
                "mtime": 1234567890.0,
                "size": 100,
            }
            mock_processor_instance.compute_file_hash.return_value = "abc123"
            mock_processor.return_value = mock_processor_instance

            # Setup mock metadata index
            mock_metadata_index_instance = MagicMock()
            mock_metadata_index.return_value = mock_metadata_index_instance

            api = CodeRAGAPI(reranker_enabled=False)

            yield {
                "api": api,
                "embedding": mock_embedding,
                "database": mock_chroma_instance,
                "processor": mock_processor_instance,
                "processor_class": mock_processor,
                "metadata_index": mock_metadata_index_instance,
                "metadata_index_class": mock_metadata_index,
                "config": mock_config_instance,
            }

    def test_index_codebase_validates_path_exists(self, mock_api_for_indexing):
        """Test that index_codebase raises error for non-existent path."""
        api = mock_api_for_indexing["api"]

        with pytest.raises(ValueError, match="Path does not exist"):
            api.index_codebase("/nonexistent/path")

    def test_index_codebase_validates_path_is_directory(self, mock_api_for_indexing):
        """Test that index_codebase raises error for non-directory path."""
        api = mock_api_for_indexing["api"]

        with tempfile.NamedTemporaryFile() as tmp_file:
            with pytest.raises(ValueError, match="Path is not a directory"):
                api.index_codebase(tmp_file.name)

    def test_index_codebase_discovers_files(self, mock_api_for_indexing):
        """Test that index_codebase discovers files in the codebase."""
        api = mock_api_for_indexing["api"]
        mock_processor = mock_api_for_indexing["processor"]
        mock_processor_class = mock_api_for_indexing["processor_class"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            # Verify processor was created
            mock_processor_class.assert_called()
            # Verify files were discovered
            mock_processor.discover_files.assert_called_once()

    def test_index_codebase_processes_files_into_chunks(self, mock_api_for_indexing):
        """Test that index_codebase processes files into chunks."""
        api = mock_api_for_indexing["api"]
        mock_processor = mock_api_for_indexing["processor"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            mock_processor.process_file.assert_called()

    def test_index_codebase_embeds_chunks(self, mock_api_for_indexing):
        """Test that index_codebase generates embeddings for chunks."""
        api = mock_api_for_indexing["api"]
        mock_embedding = mock_api_for_indexing["embedding"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            mock_embedding.embed_batch.assert_called()

    def test_index_codebase_stores_in_database(self, mock_api_for_indexing):
        """Test that index_codebase stores embeddings in database."""
        api = mock_api_for_indexing["api"]
        mock_db = mock_api_for_indexing["database"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            mock_db.add.assert_called()
            call_args = mock_db.add.call_args
            assert "ids" in call_args.kwargs
            assert "embeddings" in call_args.kwargs
            assert "documents" in call_args.kwargs
            assert "metadatas" in call_args.kwargs

    def test_index_codebase_returns_chunk_count(self, mock_api_for_indexing):
        """Test that index_codebase returns the number of chunks processed."""
        api = mock_api_for_indexing["api"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            total_chunks = api.index_codebase(tmp_dir, collection_name="test")

            assert total_chunks == 1  # Based on mock setup

    def test_index_codebase_returns_zero_for_empty_codebase(
        self, mock_api_for_indexing
    ):
        """Test that index_codebase returns 0 when no files are found."""
        api = mock_api_for_indexing["api"]
        mock_processor = mock_api_for_indexing["processor"]
        mock_processor.discover_files.return_value = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            total_chunks = api.index_codebase(tmp_dir, collection_name="test")

            assert total_chunks == 0

    def test_index_codebase_calls_progress_callback(self, mock_api_for_indexing):
        """Test that index_codebase calls progress callback during processing."""
        api = mock_api_for_indexing["api"]
        progress_callback = MagicMock()

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(
                tmp_dir, collection_name="test", progress_callback=progress_callback
            )

            progress_callback.assert_called()
            # Should be called with (file_count, total_files, file_path)
            call_args = progress_callback.call_args[0]
            assert len(call_args) == 3

    def test_index_codebase_clears_embedding_cache(self, mock_api_for_indexing):
        """Test that index_codebase clears VRAM cache after indexing."""
        api = mock_api_for_indexing["api"]
        mock_embedding = mock_api_for_indexing["embedding"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            mock_embedding.clear_cache.assert_called()

    def test_index_codebase_batches_chunks(self, mock_api_for_indexing):
        """Test that index_codebase processes chunks in batches."""
        api = mock_api_for_indexing["api"]
        mock_processor = mock_api_for_indexing["processor"]
        mock_embedding = mock_api_for_indexing["embedding"]
        mock_config = mock_api_for_indexing["config"]

        # Configure small batch size
        mock_config.get_batch_size.return_value = 2

        # Create multiple chunks
        mock_processor.discover_files.return_value = [
            "/test/file1.py",
            "/test/file2.py",
            "/test/file3.py",
        ]
        mock_processor.process_file.side_effect = [
            [{"id": "chunk_0", "content": "content 0", "metadata": {}}],
            [{"id": "chunk_1", "content": "content 1", "metadata": {}}],
            [{"id": "chunk_2", "content": "content 2", "metadata": {}}],
        ]
        mock_embedding.embed_batch.return_value = [[0.1] * 384] * 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.index_codebase(tmp_dir, collection_name="test")

            # Should have called embed_batch multiple times due to batching
            assert mock_embedding.embed_batch.call_count >= 1


# ============================================================================
# CodeRAGAPI.incremental_reindex Tests
# ============================================================================


class TestCodeRAGAPIIncrementalReindex:
    """Tests for CodeRAGAPI.incremental_reindex method."""

    @pytest.fixture
    def mock_api_for_reindex(self):
        """Create a CodeRAGAPI instance for incremental reindex tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch("code_rag.api.FileProcessor") as mock_processor,
            patch("code_rag.api.MetadataIndex") as mock_metadata_index,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config_instance.should_exclude_tests.return_value = False
            mock_config_instance.should_include_file_header.return_value = True
            mock_config_instance.get_additional_ignore_patterns.return_value = []
            mock_config_instance.get_chunk_size.return_value = 1500
            mock_config_instance.get_batch_size.return_value = 100
            mock_config_instance.should_verify_changes_with_hash.return_value = False
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_embedding.embed_batch.return_value = [[0.1] * 384]
            mock_embedding.clear_cache.return_value = None
            mock_st.return_value = mock_embedding

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.get_ids_by_file.return_value = ["old_chunk_1"]
            mock_chroma_instance.delete_by_ids.return_value = None
            mock_chroma.return_value = mock_chroma_instance

            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            mock_processor_instance = MagicMock()
            mock_processor_instance.discover_files.return_value = ["/test/file1.py"]
            mock_processor_instance.process_file.return_value = [
                {"id": "new_chunk_0", "content": "new content", "metadata": {}}
            ]
            mock_processor_instance.get_file_stats.return_value = {
                "mtime": 1234567890.0,
                "size": 100,
            }
            mock_processor_instance.compute_file_hash.return_value = "newhash"
            mock_processor.return_value = mock_processor_instance

            mock_metadata_index_instance = MagicMock()
            mock_metadata_index_instance.detect_changes.return_value = {
                "added": ["/test/new_file.py"],
                "modified": ["/test/modified_file.py"],
                "deleted": ["/test/deleted_file.py"],
                "unchanged": ["/test/unchanged_file.py"],
            }
            mock_metadata_index.return_value = mock_metadata_index_instance

            api = CodeRAGAPI(reranker_enabled=False)
            api._metadata_indices["test_collection"] = mock_metadata_index_instance

            yield {
                "api": api,
                "database": mock_chroma_instance,
                "processor": mock_processor_instance,
                "metadata_index": mock_metadata_index_instance,
            }

    def test_incremental_reindex_detects_changes(self, mock_api_for_reindex):
        """Test that incremental_reindex detects file changes."""
        api = mock_api_for_reindex["api"]
        mock_metadata_index = mock_api_for_reindex["metadata_index"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.incremental_reindex(tmp_dir, collection_name="test_collection")

            mock_metadata_index.detect_changes.assert_called()

    def test_incremental_reindex_deletes_removed_file_chunks(
        self, mock_api_for_reindex
    ):
        """Test that incremental_reindex deletes chunks for removed files."""
        api = mock_api_for_reindex["api"]
        mock_db = mock_api_for_reindex["database"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.incremental_reindex(tmp_dir, collection_name="test_collection")

            # Should get chunk IDs for deleted file
            mock_db.get_ids_by_file.assert_any_call("/test/deleted_file.py")
            # Should delete those chunks
            mock_db.delete_by_ids.assert_called()

    def test_incremental_reindex_returns_summary(self, mock_api_for_reindex):
        """Test that incremental_reindex returns a summary dict."""
        api = mock_api_for_reindex["api"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = api.incremental_reindex(tmp_dir, collection_name="test_collection")

            assert "added_count" in result
            assert "modified_count" in result
            assert "deleted_count" in result
            assert "unchanged_count" in result
            assert "total_chunks" in result

            assert result["added_count"] == 1
            assert result["modified_count"] == 1
            assert result["deleted_count"] == 1
            assert result["unchanged_count"] == 1


# ============================================================================
# CodeRAGAPI Collection Management Tests
# ============================================================================


class TestCodeRAGAPICollectionManagement:
    """Tests for CodeRAGAPI collection management methods."""

    @pytest.fixture
    def mock_api_for_collection(self):
        """Create a CodeRAGAPI instance for collection management tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_st.return_value = mock_embedding

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.initialize.return_value = None
            mock_chroma_instance.delete_collection.return_value = None
            mock_chroma_instance.is_processed.return_value = True
            mock_chroma_instance.count.return_value = 100
            mock_chroma.return_value = mock_chroma_instance

            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            api = CodeRAGAPI(reranker_enabled=False)

            yield {
                "api": api,
                "database": mock_chroma_instance,
                "embedding": mock_embedding,
            }

    def test_initialize_collection_creates_collection(self, mock_api_for_collection):
        """Test that initialize_collection creates a new collection."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        api.initialize_collection("my_collection")

        mock_db.initialize.assert_called()
        call_args = mock_db.initialize.call_args
        assert call_args[0][0] == "my_collection"

    def test_initialize_collection_with_force_reindex(self, mock_api_for_collection):
        """Test that initialize_collection with force_reindex deletes existing collection."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        api.initialize_collection("my_collection", force_reindex=True)

        mock_db.delete_collection.assert_called_with("my_collection")
        mock_db.initialize.assert_called()

    def test_initialize_collection_handles_dimension_mismatch(
        self, mock_api_for_collection
    ):
        """Test that initialize_collection handles dimension mismatch."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        # Simulate dimension mismatch - database returns stored model name
        mock_db.initialize.return_value = "different-model"

        with patch.object(api, "_create_embedding_model") as mock_create:
            mock_new_embedding = MagicMock()
            mock_new_embedding.get_embedding_dimension.return_value = 768
            mock_create.return_value = mock_new_embedding

            result = api.initialize_collection("my_collection")

            # Should return the stored model name
            assert result == "different-model"
            # Should reload embedding model
            mock_create.assert_called_with("different-model", lazy_load=False)

    def test_is_processed_delegates_to_database(self, mock_api_for_collection):
        """Test that is_processed delegates to database."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        result = api.is_processed()

        mock_db.is_processed.assert_called()
        assert result is True

    def test_is_processed_handles_exception(self, mock_api_for_collection):
        """Test that is_processed returns False on exception."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]
        mock_db.is_processed.side_effect = RuntimeError("Database error")

        result = api.is_processed()

        assert result is False

    def test_count_delegates_to_database(self, mock_api_for_collection):
        """Test that count delegates to database."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        result = api.count()

        mock_db.count.assert_called()
        assert result == 100

    def test_count_handles_exception(self, mock_api_for_collection):
        """Test that count returns 0 on exception."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]
        mock_db.count.side_effect = RuntimeError("Database error")

        result = api.count()

        assert result == 0

    def test_close_closes_database(self, mock_api_for_collection):
        """Test that close method closes database connection."""
        api = mock_api_for_collection["api"]
        mock_db = mock_api_for_collection["database"]

        api.close()

        mock_db.close.assert_called()


# ============================================================================
# CodeRAGAPI.ensure_indexed Tests
# ============================================================================


class TestCodeRAGAPIEnsureIndexed:
    """Tests for CodeRAGAPI.ensure_indexed method."""

    @pytest.fixture
    def mock_api_for_ensure_indexed(self):
        """Create a CodeRAGAPI instance for ensure_indexed tests."""
        with (
            patch("code_rag.api.SentenceTransformerEmbedding") as mock_st,
            patch("code_rag.api.ChromaDatabase") as mock_chroma,
            patch("code_rag.api.CrossEncoderReranker") as mock_reranker,
            patch("code_rag.api.Config") as mock_config,
            patch("code_rag.api.FileProcessor") as mock_processor,
            patch("code_rag.api.MetadataIndex") as mock_metadata_index,
        ):

            mock_config_instance = MagicMock()
            mock_config_instance.get_database_path.return_value = "/tmp/test-db"
            mock_config_instance.is_shared_server_enabled.return_value = False
            mock_config_instance.get_shared_server_port.return_value = 8199
            mock_config_instance.get_reranker_model.return_value = (
                "jinaai/jina-reranker-v3"
            )
            mock_config_instance.get_model_idle_timeout.return_value = 1800
            mock_config_instance.should_exclude_tests.return_value = False
            mock_config_instance.should_include_file_header.return_value = True
            mock_config_instance.get_additional_ignore_patterns.return_value = []
            mock_config_instance.get_chunk_size.return_value = 1500
            mock_config_instance.get_batch_size.return_value = 100
            mock_config_instance.should_verify_changes_with_hash.return_value = False
            mock_config_instance.get_reindex_debounce_minutes.return_value = 30
            mock_config.return_value = mock_config_instance

            mock_embedding = MagicMock()
            mock_embedding.get_embedding_dimension.return_value = 384
            mock_embedding.embed_batch.return_value = [[0.1] * 384]
            mock_embedding.clear_cache.return_value = None
            mock_st.return_value = mock_embedding

            mock_chroma_instance = MagicMock()
            mock_chroma_instance.initialize.return_value = None
            mock_chroma_instance.is_processed.return_value = False
            mock_chroma_instance.count.return_value = 0
            mock_chroma.return_value = mock_chroma_instance

            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            mock_processor_instance = MagicMock()
            mock_processor_instance.discover_files.return_value = ["/test/file.py"]
            mock_processor_instance.process_file.return_value = [
                {"id": "chunk_0", "content": "test content", "metadata": {}}
            ]
            mock_processor_instance.get_file_stats.return_value = {
                "mtime": 1234567890.0,
                "size": 100,
            }
            mock_processor.return_value = mock_processor_instance

            mock_metadata_index_instance = MagicMock()
            mock_metadata_index_instance.should_reindex.return_value = False
            mock_metadata_index.return_value = mock_metadata_index_instance

            api = CodeRAGAPI(reranker_enabled=False)

            yield {
                "api": api,
                "database": mock_chroma_instance,
                "processor": mock_processor_instance,
                "metadata_index": mock_metadata_index_instance,
                "config": mock_config_instance,
            }

    def test_ensure_indexed_validates_path_exists(self, mock_api_for_ensure_indexed):
        """Test that ensure_indexed returns error for non-existent path."""
        api = mock_api_for_ensure_indexed["api"]

        result = api.ensure_indexed("/nonexistent/path")

        assert result["success"] is False
        assert "does not exist" in result["error"]

    def test_ensure_indexed_validates_path_is_directory(
        self, mock_api_for_ensure_indexed
    ):
        """Test that ensure_indexed returns error for non-directory."""
        api = mock_api_for_ensure_indexed["api"]

        with tempfile.NamedTemporaryFile() as tmp_file:
            result = api.ensure_indexed(tmp_file.name)

            assert result["success"] is False
            assert "not a directory" in result["error"]

    def test_ensure_indexed_uses_session_cache(self, mock_api_for_ensure_indexed):
        """Test that ensure_indexed uses session cache for already indexed paths."""
        api = mock_api_for_ensure_indexed["api"]
        mock_db = mock_api_for_ensure_indexed["database"]
        mock_db.count.return_value = 50

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Add to session cache
            api._indexed_paths.add(str(Path(tmp_dir).resolve()))

            result = api.ensure_indexed(tmp_dir)

            assert result["success"] is True
            assert result["already_indexed"] is True

    def test_ensure_indexed_auto_generates_collection_name(
        self, mock_api_for_ensure_indexed
    ):
        """Test that ensure_indexed auto-generates collection name from path."""
        api = mock_api_for_ensure_indexed["api"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.ensure_indexed(tmp_dir)

            # Should have set an active collection
            assert api._active_collection is not None
            assert api._active_collection.startswith("codebase_")

    def test_ensure_indexed_uses_provided_collection_name(
        self, mock_api_for_ensure_indexed
    ):
        """Test that ensure_indexed uses provided collection name."""
        api = mock_api_for_ensure_indexed["api"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            api.ensure_indexed(tmp_dir, collection_name="custom_collection")

            assert api._active_collection == "custom_collection"

    def test_ensure_indexed_returns_success_after_indexing(
        self, mock_api_for_ensure_indexed
    ):
        """Test that ensure_indexed returns success after indexing."""
        api = mock_api_for_ensure_indexed["api"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = api.ensure_indexed(tmp_dir, collection_name="test")

            assert result["success"] is True
            assert "total_chunks" in result

    def test_ensure_indexed_with_force_reindex(self, mock_api_for_ensure_indexed):
        """Test that ensure_indexed with force_reindex reindexes even if cached."""
        api = mock_api_for_ensure_indexed["api"]
        mock_db = mock_api_for_ensure_indexed["database"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            # First, mark as already indexed
            api._indexed_paths.add(str(Path(tmp_dir).resolve()))
            mock_db.count.return_value = 100

            # Force reindex
            result = api.ensure_indexed(
                tmp_dir, collection_name="test", force_reindex=True
            )

            assert result["success"] is True
            # Should NOT use the cache (already_indexed should be False because we're reindexing)
            assert result.get("already_indexed") is False

    def test_ensure_indexed_skips_reindex_with_debounce(
        self, mock_api_for_ensure_indexed
    ):
        """Test that ensure_indexed respects debounce period."""
        api = mock_api_for_ensure_indexed["api"]
        mock_db = mock_api_for_ensure_indexed["database"]
        mock_metadata_index = mock_api_for_ensure_indexed["metadata_index"]

        # Setup: database already has data
        mock_db.is_processed.return_value = True
        mock_db.count.return_value = 100
        # Debounce is active (should_reindex returns False)
        mock_metadata_index.should_reindex.return_value = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            api._metadata_indices["test"] = mock_metadata_index

            result = api.ensure_indexed(tmp_dir, collection_name="test")

            assert result["success"] is True
            assert result["already_indexed"] is True
            assert result.get("debounce_active") is True
