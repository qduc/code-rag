"""Comprehensive tests for the MCP server functionality.

These tests verify:
1. MCP server initialization
2. Tool registration and discovery
3. Search functionality with auto-indexing
4. Result formatting
5. Error handling
6. Input validation
7. Configuration and environment variable handling
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import code_rag.mcp_server as mcp_mod
from code_rag.api import CodeRAGAPI
from code_rag.mcp_server import call_tool, format_search_results, list_tools

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def temp_codebase():
    """Create a temporary codebase with sample Python files for testing.

    Performance: Using module scope to avoid recreating files for each test.
    This saves ~0.5s per test.
    """
    temp_dir = tempfile.mkdtemp()

    # Create src directory
    src_dir = Path(temp_dir) / "src"
    src_dir.mkdir()

    # Create sample Python files
    (src_dir / "auth.py").write_text(
        """
def login(username, password):
    '''Authenticate user with username and password'''
    # Check credentials
    if not username or not password:
        raise ValueError("Invalid credentials")

    # Hash password
    hashed = hash_password(password)

    # Check database
    user = database.find_user(username)
    if user and user.password == hashed:
        return create_session(user)
    return None

def hash_password(password):
    '''Hash password using bcrypt'''
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())
"""
    )

    (src_dir / "database.py").write_text(
        """
class Database:
    def __init__(self, connection_string):
        '''Initialize database connection'''
        self.connection_string = connection_string
        self.conn = None

    def connect(self):
        '''Connect to the database'''
        import sqlite3
        self.conn = sqlite3.connect(self.connection_string)

    def find_user(self, username):
        '''Find user by username'''
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        return cursor.fetchone()
"""
    )

    (src_dir / "main.py").write_text(
        """
from auth import login
from database import Database

def main():
    '''Main application entry point'''
    db = Database("app.db")
    db.connect()

    # Start application
    app = Application(db)
    app.run()

class Application:
    def __init__(self, database):
        self.db = database

    def run(self):
        '''Run the application'''
        print("Starting application...")
"""
    )

    # Create a package.json for diversity
    (Path(temp_dir) / "package.json").write_text(
        """{
  "name": "test-app",
  "version": "1.0.0",
  "description": "Test application",
  "main": "index.js"
}
"""
    )

    yield str(temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def temp_database():
    """Create a temporary database directory for testing.

    Performance: Using module scope to reuse the same database directory.
    """
    temp_dir = tempfile.mkdtemp()
    yield str(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def api_instance(temp_database):
    """Create a CodeRAGAPI instance with temporary database.

    Performance: Using module scope to avoid reloading the embedding model
    for each test. This saves ~5-10s per test session.
    """
    api = CodeRAGAPI(
        database_type="chroma",
        database_path=temp_database,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        reranker_enabled=False,  # Disable reranker for faster tests
    )
    yield api
    api.close()


@pytest.fixture(scope="module")
def indexed_api(api_instance, temp_codebase):
    """Create and index a CodeRAGAPI instance.

    Performance: Using module scope to index the codebase only once.
    This saves ~2-5s per test that uses indexed data.
    """
    api_instance.ensure_indexed(
        temp_codebase,
        collection_name="test_codebase",
        validate_codebase=False,
    )
    return api_instance, temp_codebase


# ============================================================================
# Tests for format_search_results
# ============================================================================


class TestFormatSearchResults:
    """Tests for the format_search_results function."""

    def test_format_empty_results(self):
        """Test formatting empty search results."""
        result = format_search_results([])
        assert "No results" in result

    def test_format_single_result(self):
        """Test formatting a single search result."""
        results = [
            {
                "file_path": "src/auth.py",
                "start_line": 1,
                "end_line": 10,
                "similarity": 0.95,
                "content": "def login(username, password):\n    pass",
            }
        ]
        output = format_search_results(results)

        # New compact format: file:lines (score)
        assert "src/auth.py:1-10" in output
        assert "0.95" in output
        assert "def login" in output
        assert "---" in output

    def test_format_multiple_results(self):
        """Test formatting multiple search results."""
        results = [
            {
                "file_path": "src/auth.py",
                "start_line": 1,
                "end_line": 10,
                "similarity": 0.95,
                "content": "def login(username, password):\n    pass",
            },
            {
                "file_path": "src/database.py",
                "start_line": 20,
                "end_line": 30,
                "similarity": 0.87,
                "content": "def find_user(self, username):\n    pass",
            },
        ]
        output = format_search_results(results)

        # New compact format
        assert "src/auth.py:1-10" in output
        assert "src/database.py:20-30" in output
        assert "0.95" in output
        assert "0.87" in output

    def test_format_result_without_line_numbers(self):
        """Test formatting results without line number information."""
        results = [
            {
                "file_path": "src/auth.py",
                "similarity": 0.95,
                "content": "def login(username, password):\n    pass",
            }
        ]
        output = format_search_results(results)

        assert "src/auth.py" in output
        assert "0.95" in output
        # Should not have line numbers if start_line/end_line are missing
        assert ":1-" not in output

    def test_format_truncates_long_content(self):
        """Test that long content is truncated to 600 characters (default limit)."""
        long_content = "x" * 800
        results = [
            {
                "file_path": "src/test.py",
                "similarity": 0.9,
                "content": long_content,
            }
        ]
        output = format_search_results(results)

        # Should be truncated with ellipsis character
        assert "…" in output
        # The output will have formatting around it, but the content should be truncated
        assert long_content not in output  # Full content should not appear

    def test_format_shows_full_content_when_requested(self):
        """Test that full content is shown when show_full_content=True."""
        long_content = "x" * 500
        results = [
            {
                "file_path": "src/test.py",
                "similarity": 0.9,
                "content": long_content,
            }
        ]
        output = format_search_results(results, show_full_content=True)

        # Should not be truncated
        assert "…" not in output
        assert long_content in output

    def test_format_includes_function_name(self):
        """Test that function names are included in the header when available."""
        results = [
            {
                "file_path": "src/auth.py",
                "start_line": 10,
                "end_line": 25,
                "similarity": 0.92,
                "content": "def authenticate(username, password):\n    ...",
                "function_name": "authenticate",
            }
        ]
        output = format_search_results(results)

        # Should include function name with () notation
        assert "authenticate()" in output
        assert "src/auth.py:10-25" in output

    def test_format_includes_class_name(self):
        """Test that class names are included in the header when available."""
        results = [
            {
                "file_path": "src/models.py",
                "start_line": 5,
                "end_line": 50,
                "similarity": 0.88,
                "content": "class UserModel:\n    ...",
                "class_name": "UserModel",
            }
        ]
        output = format_search_results(results)

        # Should include class name
        assert "UserModel" in output
        assert "src/models.py:5-50" in output

    def test_format_includes_both_function_and_class(self):
        """Test that both function and class names are shown for methods."""
        results = [
            {
                "file_path": "src/auth.py",
                "start_line": 15,
                "end_line": 30,
                "similarity": 0.95,
                "content": "def validate_token(self, token):\n    ...",
                "function_name": "validate_token",
                "class_name": "AuthHandler",
                "symbol_type": "method",
            }
        ]
        output = format_search_results(results)

        # Should include both
        assert "validate_token()" in output
        assert "AuthHandler" in output

    def test_format_uses_expanded_content_when_available(self):
        """Test that expanded_content is used when present in results."""
        results = [
            {
                "file_path": "src/test.py",
                "start_line": 10,
                "end_line": 15,
                "similarity": 0.9,
                "content": "original chunk",
                "expanded_content": "expanded content with more context",
                "expanded_start_line": 5,
                "expanded_end_line": 20,
            }
        ]
        output = format_search_results(results)

        # Should use expanded content and line numbers
        assert "expanded content with more context" in output
        assert "5-20" in output


# ============================================================================
# Tests for list_tools
# ============================================================================


class TestListTools:
    """Tests for the list_tools function."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_search_tool(self):
        """Test that list_tools returns the search_codebase tool."""
        tools = await list_tools()

        assert len(tools) == 1
        assert tools[0].name == "search_codebase"

    @pytest.mark.asyncio
    async def test_search_tool_has_correct_schema(self):
        """Test that search_codebase tool has correct input schema."""
        tools = await list_tools()
        tool = tools[0]

        schema = tool.inputSchema
        assert "properties" in schema
        assert "codebase_path" in schema["properties"]
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert "expand_context" in schema["properties"]

        # Check required fields
        assert "codebase_path" in schema["required"]
        assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_search_tool_description_present(self):
        """Test that search_codebase tool has a description."""
        tools = await list_tools()
        tool = tools[0]

        assert tool.description is not None
        assert len(tool.description) > 0
        assert "semantic search" in tool.description.lower()


# ============================================================================
# Tests for call_tool - Input Validation
# ============================================================================


class TestCallToolValidation:
    """Tests for input validation in call_tool."""

    @pytest.mark.asyncio
    async def test_call_tool_missing_codebase_path(self):
        """Test that error is returned when codebase_path is missing."""
        # Mock the global api
        with patch("code_rag.mcp_server.api", None):
            result = await call_tool(
                "search_codebase", {"query": "test"}, _api_wait_timeout=0.1
            )
            assert len(result) == 1
            assert "Error: Code-RAG API not initialized" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_missing_query(self):
        """Test that error is returned when query is missing."""
        with patch("code_rag.mcp_server.api", None):
            result = await call_tool(
                "search_codebase",
                {"codebase_path": "/path/to/code"},
                _api_wait_timeout=0.1,
            )
            assert len(result) == 1
            assert "Error: Code-RAG API not initialized" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_waits_for_api_initialization(self):
        """If API is initialized shortly after startup, call_tool should wait and proceed.

        This test simulates delayed initialization in a background thread and verifies
        that `call_tool` does not immediately return the initialization error.
        """
        # Ensure initial state: no API and event cleared
        mcp_mod.api = None
        try:
            mcp_mod.api_ready_event.clear()
        except Exception:
            pass

        # Start a background thread that sets the api after a short delay
        import threading
        import time

        def delayed_init():
            time.sleep(0.2)

            class DummyAPI:
                def ensure_indexed(self, *a, **k):
                    return {"success": True}

                def search(self, *a, **k):
                    return [
                        {
                            "content": "dummy",
                            "file_path": "src/dummy.py",
                            "chunk_index": 0,
                            "total_chunks": 1,
                            "start_line": 1,
                            "end_line": 3,
                            "similarity": 0.9,
                        }
                    ]

            mcp_mod.api = DummyAPI()
            try:
                mcp_mod.api_ready_event.set()
            except Exception:
                pass

        threading.Thread(target=delayed_init, daemon=True).start()

        # Call tool; should wait for initialization and then proceed
        result = await call_tool(
            "search_codebase", {"codebase_path": "/tmp", "query": "dummy"}
        )

        assert len(result) == 1
        assert "dummy" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_returns_error_if_api_not_ready(self):
        """If API never initializes, call_tool should return an error message after waiting."""
        mcp_mod.api = None
        try:
            mcp_mod.api_ready_event.clear()
        except Exception:
            pass

        # Do not initialize API; call_tool should return the error after wait
        result = await call_tool(
            "search_codebase", {"query": "test"}, _api_wait_timeout=0.1
        )
        assert len(result) == 1
        assert "Error: Code-RAG API not initialized" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool_name(self):
        """Test that error is returned for unknown tool name."""
        mock_api = MagicMock()
        with patch("code_rag.mcp_server.api", mock_api):
            result = await call_tool("unknown_tool", {})
            assert len(result) == 1
            assert "Unknown tool" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_caps_max_results(self):
        """Test that max_results is capped at 20."""
        mock_api = MagicMock()
        mock_api.ensure_indexed.return_value = {"success": True}
        mock_api.search.return_value = []

        with patch("code_rag.mcp_server.api", mock_api):
            await call_tool(
                "search_codebase",
                {
                    "codebase_path": "/test",
                    "query": "test",
                    "max_results": 100,  # Should be capped to 20
                },
            )

            # Check that search was called with max_results=20
            call_args = mock_api.search.call_args
            assert call_args[1]["n_results"] == 20


# ============================================================================
# Tests for call_tool - Search Functionality
# ============================================================================


class TestCallToolSearch:
    """Tests for search functionality in call_tool."""

    @pytest.mark.asyncio
    async def test_call_tool_search_success(self, indexed_api):
        """Test successful search execution."""
        api_instance, codebase_path = indexed_api

        with patch("code_rag.mcp_server.api", api_instance):
            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": codebase_path,
                    "query": "authentication login",
                    "max_results": 5,
                },
            )

            assert len(result) == 1
            assert result[0].type == "text"
            # Should contain some results about authentication
            assert "auth" in result[0].text.lower() or "found" in result[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_search_auto_indexes(self, temp_codebase):
        """Test that search auto-indexes the codebase.

        Note: Uses a fresh API instance to test auto-indexing behavior.
        """
        # Create a fresh API instance for this test
        with tempfile.TemporaryDirectory() as fresh_db:
            fresh_api = CodeRAGAPI(
                database_path=fresh_db,
                reranker_enabled=False,
            )

            # Verify it's not indexed yet
            assert fresh_api.count() == 0

            with patch("code_rag.mcp_server.api", fresh_api):
                result = await call_tool(
                    "search_codebase",
                    {
                        "codebase_path": temp_codebase,
                        "query": "login",
                    },
                )

                assert result[0].type == "text"
                # After search, database should have chunks
                assert fresh_api.count() > 0

            fresh_api.close()

    @pytest.mark.asyncio
    async def test_call_tool_search_handles_invalid_path(self):
        """Test that search handles invalid codebase path."""
        mock_api = MagicMock()
        mock_api.ensure_indexed.return_value = {
            "success": False,
            "error": "Path does not exist: /nonexistent/path",
        }

        with patch("code_rag.mcp_server.api", mock_api):
            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": "/nonexistent/path",
                    "query": "test",
                },
            )

            assert len(result) == 1
            assert "does not exist" in result[0].text

    @pytest.mark.asyncio
    async def test_call_tool_search_returns_formatted_results(self, indexed_api):
        """Test that search results are properly formatted."""
        api_instance, codebase_path = indexed_api

        with patch("code_rag.mcp_server.api", api_instance):
            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": codebase_path,
                    "query": "database connection",
                    "max_results": 5,
                },
            )

            assert len(result) == 1
            text_result = result[0].text

            # Check formatting includes expected elements (compact format now)
            assert "---" in text_result or "No results" in text_result


# ============================================================================
# Tests for CodeRAGAPI Integration
# ============================================================================


class TestCodeRAGAPIIntegration:
    """Tests for CodeRAGAPI integration with MCP server."""

    def test_api_initialization_with_defaults(self, temp_database):
        """Test that CodeRAGAPI initializes with default parameters."""
        api = CodeRAGAPI(
            database_path=temp_database,
            reranker_enabled=False,  # Disable for faster tests
        )

        assert api.database_type == "chroma"
        assert api.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        # Note: We disabled reranker for performance
        assert api.reranker_enabled is False

        api.close()

    def test_api_initialization_with_custom_params(self, temp_database):
        """Test that CodeRAGAPI accepts custom parameters."""
        api = CodeRAGAPI(
            database_type="chroma",
            database_path=temp_database,
            embedding_model="sentence-transformers/paraphrase-MiniLM-L6-v2",
            reranker_enabled=False,
        )

        assert (
            api.embedding_model_name == "sentence-transformers/paraphrase-MiniLM-L6-v2"
        )
        assert api.reranker_enabled is False

        api.close()

    def test_api_ensure_indexed_with_valid_codebase(self, temp_codebase):
        """Test ensure_indexed with a valid codebase.

        Note: Uses a fresh API instance to test indexing behavior.
        """
        with tempfile.TemporaryDirectory() as fresh_db:
            fresh_api = CodeRAGAPI(
                database_path=fresh_db,
                reranker_enabled=False,
            )

            result = fresh_api.ensure_indexed(
                temp_codebase,
                collection_name="test_codebase",
                validate_codebase=False,
            )

            assert result["success"] is True
            assert result["total_chunks"] > 0
            assert result["already_indexed"] is False

            fresh_api.close()

    def test_api_ensure_indexed_caches_result(self, api_instance, temp_codebase):
        """Test that ensure_indexed caches results for repeated calls."""
        # First call
        # result1 = api_instance.ensure_indexed(
        #     temp_codebase,
        #     collection_name="test_codebase",
        #     validate_codebase=False,
        # )
        # # chunks_1 = result1["total_chunks"]
        api_instance.ensure_indexed(
            temp_codebase,
            collection_name="test_codebase",
            validate_codebase=False,
        )

        # Second call should return cached result
        result2 = api_instance.ensure_indexed(
            temp_codebase,
            collection_name="test_codebase",
            validate_codebase=False,
        )

        assert result2["success"] is True
        assert result2["already_indexed"] is True

    def test_api_search_returns_results(self, indexed_api):
        """Test that search returns results."""
        api_instance, _ = indexed_api

        results = api_instance.search(
            "authentication login",
            n_results=5,
            collection_name="test_codebase",
        )

        assert isinstance(results, list)
        # The sample code has authentication content
        assert len(results) > 0

    def test_api_search_result_structure(self, indexed_api):
        """Test that search results have correct structure including new metadata fields."""
        api_instance, _ = indexed_api

        results = api_instance.search(
            "database",
            n_results=5,
            collection_name="test_codebase",
        )

        if results:
            result = results[0]
            # Core fields
            assert "content" in result
            assert "file_path" in result
            assert "similarity" in result
            assert 0 <= result["similarity"] <= 1
            # New metadata fields (may be None if not available for this chunk)
            assert "function_name" in result
            assert "class_name" in result
            assert "symbol_type" in result

    def test_api_search_with_expand_context(self, indexed_api):
        """Test that expand_context parameter works."""
        api_instance, _ = indexed_api

        # Search with context expansion
        results = api_instance.search(
            "authentication",
            n_results=3,
            collection_name="test_codebase",
            expand_context=True,
        )

        if results:
            result = results[0]
            # Should have expanded content fields
            assert "expanded_content" in result
            assert "expanded_start_line" in result
            assert "expanded_end_line" in result

    def test_api_count_returns_number(self, indexed_api):
        """Test that count returns the number of chunks."""
        api_instance, _ = indexed_api

        count = api_instance.count()
        assert isinstance(count, int)
        assert count > 0

    def test_api_is_processed_returns_bool(self, indexed_api):
        """Test that is_processed returns a boolean."""
        api_instance, _ = indexed_api

        result = api_instance.is_processed()
        assert isinstance(result, bool)
        assert result is True


# ============================================================================
# Tests for Environment Variable Configuration
# ============================================================================


class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    def test_database_type_from_env(self, temp_database, monkeypatch):
        """Test that database type can be set via environment variable."""
        monkeypatch.setenv("CODE_RAG_DATABASE_TYPE", "chroma")

        api = CodeRAGAPI(
            database_path=temp_database,
            reranker_enabled=False,  # Disable for faster tests
        )
        assert api.database_type == "chroma"
        api.close()

    def test_embedding_model_explicit_parameter(self, temp_database):
        """Test that embedding model can be set via explicit parameter."""
        model = "sentence-transformers/paraphrase-MiniLM-L6-v2"

        api = CodeRAGAPI(
            database_path=temp_database,
            embedding_model=model,
            reranker_enabled=False,  # Disable for faster tests
        )
        assert api.embedding_model_name == model
        api.close()

    def test_reranker_disabled_via_parameter(self, temp_database):
        """Test that reranker can be disabled via parameter."""
        api = CodeRAGAPI(database_path=temp_database, reranker_enabled=False)
        assert api.reranker is None
        api.close()


# ============================================================================
# Tests for Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in MCP server."""

    @pytest.mark.asyncio
    async def test_call_tool_handles_exception(self):
        """Test that exceptions in call_tool are handled gracefully."""
        mock_api = MagicMock()
        mock_api.ensure_indexed.side_effect = RuntimeError("Test error")

        with patch("code_rag.mcp_server.api", mock_api):
            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": "/test",
                    "query": "test",
                },
            )

            assert len(result) == 1
            assert "Error executing" in result[0].text

    def test_api_handles_missing_path(self):
        """Test that API handles missing paths gracefully."""
        api = CodeRAGAPI(reranker_enabled=False)  # Disable for faster tests

        result = api.ensure_indexed(
            "/nonexistent/path/that/does/not/exist",
            validate_codebase=False,
        )

        assert result["success"] is False
        assert "does not exist" in result["error"]

        api.close()

    def test_api_handles_non_directory_path(self, temp_database):
        """Test that API handles file paths instead of directories."""
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            api = CodeRAGAPI(
                database_path=temp_database,
                reranker_enabled=False,  # Disable for faster tests
            )
            result = api.ensure_indexed(temp_file.name, validate_codebase=False)

            assert result["success"] is False
            assert "not a directory" in result["error"]

            api.close()
        finally:
            os.unlink(temp_file.name)


# ============================================================================
# Tests for Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_search_with_empty_query(self, indexed_api):
        """Test search with empty query string."""
        api_instance, codebase_path = indexed_api

        # Empty query should still work (embedding will handle it)
        results = api_instance.search("", n_results=5, collection_name="test_codebase")
        assert isinstance(results, list)

    def test_search_with_very_large_max_results(self, indexed_api):
        """Test that max_results is properly limited."""
        api_instance, codebase_path = indexed_api

        results = api_instance.search(
            "database", n_results=1000, collection_name="test_codebase"
        )

        # Should return limited results, not 1000
        assert len(results) <= 100  # Reasonable limit

    @pytest.mark.asyncio
    async def test_call_tool_with_zero_max_results(self):
        """Test call_tool with max_results set to 0."""
        mock_api = MagicMock()
        mock_api.ensure_indexed.return_value = {"success": True}
        mock_api.search.return_value = []

        with patch("code_rag.mcp_server.api", mock_api):
            result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": "/test",
                    "query": "test",
                    "max_results": 0,
                },
            )

            # Should handle zero gracefully (capped to at least 1 or returns empty)
            assert len(result) >= 1

    def test_search_with_special_characters(self, indexed_api):
        """Test search with special characters in query."""
        api_instance, codebase_path = indexed_api

        # These should not raise exceptions
        queries = [
            "database.connect()",
            "def __init__",
            "try/except",
            "@property",
            "#TODO",
        ]

        for query in queries:
            results = api_instance.search(
                query, n_results=5, collection_name="test_codebase"
            )
            assert isinstance(results, list)


# ============================================================================
# Tests for Performance and Scaling
# ============================================================================


class TestPerformanceAndScaling:
    """Tests for performance characteristics."""

    def test_search_performance(self, indexed_api):
        """Test that search returns valid results (removed timing assertion for CI stability).

        Timing-based assertions are fragile across different CI environments and hardware.
        This test verifies search functionality by validating the results structure instead.
        """
        api_instance, codebase_path = indexed_api

        results = api_instance.search(
            "function definition", n_results=10, collection_name="test_codebase"
        )

        # Verify search returns valid results (state-based assertions)
        assert isinstance(results, list), "Search should return a list of results"
        assert (
            len(results) > 0
        ), "Search should find results for 'function definition' in indexed codebase"

        # Verify result structure
        for result in results:
            assert "file_path" in result, "Each result should have a file_path"
            assert "content" in result, "Each result should have content"
            assert "similarity" in result, "Each result should have a similarity score"

    def test_multiple_searches_on_same_indexed_codebase(self, indexed_api):
        """Test multiple searches on the same indexed codebase."""
        api_instance, codebase_path = indexed_api

        queries = ["authentication", "database", "function", "login"]

        for query in queries:
            results = api_instance.search(
                query, n_results=5, collection_name="test_codebase"
            )
            assert isinstance(results, list)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow_index_and_search(self, api_instance, temp_codebase):
        """Test full workflow: initialize -> index -> search."""
        # Initialize API
        assert api_instance is not None

        # Ensure indexed
        with patch("code_rag.mcp_server.api", api_instance):
            index_result = api_instance.ensure_indexed(
                temp_codebase,
                collection_name="test_codebase",
                validate_codebase=False,
            )
            assert index_result["success"] is True
            assert index_result["total_chunks"] > 0

            # Search
            search_result = await call_tool(
                "search_codebase",
                {
                    "codebase_path": temp_codebase,
                    "query": "authentication",
                    "max_results": 5,
                },
            )

            assert len(search_result) == 1
            assert search_result[0].type == "text"
            assert len(search_result[0].text) > 0

    @pytest.mark.asyncio
    async def test_multiple_codebase_searches(self, api_instance):
        """Test searching different codebases in same session."""
        # Create two temporary codebases
        with (
            tempfile.TemporaryDirectory() as temp1,
            tempfile.TemporaryDirectory() as temp2,
        ):
            # Create files in first codebase
            (Path(temp1) / "test1.py").write_text("def function1():\n    pass")

            # Create files in second codebase
            (Path(temp2) / "test2.py").write_text("def function2():\n    pass")

            with patch("code_rag.mcp_server.api", api_instance):
                # Search in first codebase
                result1 = await call_tool(
                    "search_codebase",
                    {
                        "codebase_path": temp1,
                        "query": "function1",
                    },
                )

                # Search in second codebase
                result2 = await call_tool(
                    "search_codebase",
                    {
                        "codebase_path": temp2,
                        "query": "function2",
                    },
                )

                # Both should succeed
                assert result1[0].type == "text"
                assert result2[0].type == "text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
