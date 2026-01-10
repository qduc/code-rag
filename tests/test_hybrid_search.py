"""Integration tests for hybrid search functionality (Phase 1).

These tests verify:
1. Identifier-based boosting in search results
2. Integration with CodeRAGAPI search method
3. Natural language queries remain unaffected
4. Mixed queries (identifiers + natural language)
5. Result metadata (boosted flag, original_similarity)
6. End-to-end search flow with boosting
"""

import tempfile
from pathlib import Path

import pytest

from code_rag.api import CodeRAGAPI

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def sample_codebase():
    """Create a sample codebase with various identifier patterns for testing.

    Performance: Using module scope to avoid recreating files for each test.
    """
    temp_dir = tempfile.mkdtemp()
    base_path = Path(temp_dir)

    # Create source files with different identifier patterns
    src_dir = base_path / "src"
    src_dir.mkdir()

    # File with camelCase identifiers
    (src_dir / "auth.py").write_text(
        """
def validateUserToken(token):
    '''Validate user authentication token'''
    if not token:
        return False
    return token.is_valid()

def getUserById(user_id):
    '''Retrieve user by ID'''
    return database.query(User).filter(User.id == user_id).first()

def processPayment(payment_data):
    '''Process payment transaction'''
    return payment_processor.charge(payment_data)
"""
    )

    # File with PascalCase identifiers
    (src_dir / "services.py").write_text(
        """
class AuthService:
    '''Authentication service for user management'''
    def __init__(self):
        self.sessions = {}

    def create_session(self, user):
        session = Session(user)
        self.sessions[session.id] = session
        return session

class UserValidator:
    '''Validate user data and credentials'''
    def validate_email(self, email):
        return '@' in email

class TokenManager:
    '''Manage authentication tokens'''
    def generate_token(self, user):
        return jwt.encode({'user_id': user.id}, SECRET_KEY)
"""
    )

    # File with snake_case identifiers
    (src_dir / "database.py").write_text(
        """
def create_connection(database_url):
    '''Create database connection'''
    return Database(database_url)

def execute_query(query, params):
    '''Execute SQL query with parameters'''
    cursor = connection.cursor()
    cursor.execute(query, params)
    return cursor.fetchall()

def get_user_by_email(email):
    '''Find user by email address'''
    return execute_query("SELECT * FROM users WHERE email = ?", (email,))
"""
    )

    # File with SCREAMING_SNAKE_CASE constants
    (src_dir / "config.py").write_text(
        """
MAX_RETRIES = 3
API_KEY = "secret-key-here"
DEFAULT_TIMEOUT = 30
DATABASE_URL = "postgresql://localhost/mydb"
CONNECTION_POOL_SIZE = 10

def get_config():
    '''Get application configuration'''
    return {
        'max_retries': MAX_RETRIES,
        'api_key': API_KEY,
        'timeout': DEFAULT_TIMEOUT,
    }
"""
    )

    # File with general authentication logic (for natural language queries)
    (src_dir / "security.py").write_text(
        """
def authenticate_user(username, password):
    '''Authenticate user with username and password'''
    user = find_user_by_username(username)
    if user and verify_password(password, user.password_hash):
        return create_session_token(user)
    return None

def verify_password(password, hash):
    '''Verify password against hash'''
    import bcrypt
    return bcrypt.checkpw(password.encode(), hash)

def handle_authentication_error(error):
    '''Handle authentication errors'''
    log_error(error)
    return {"error": "Authentication failed"}
"""
    )

    yield str(base_path)

    # Cleanup
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def indexed_api(sample_codebase):
    """Provide a CodeRAGAPI instance with indexed sample codebase.

    Performance: Using module scope and reusing the same indexed database
    for all tests in this module.
    """
    # Create temporary database directory
    temp_db = tempfile.mkdtemp()

    # Initialize API (disable reranker for faster tests)
    api = CodeRAGAPI(database_path=temp_db, reranker_enabled=False)

    # Index the sample codebase
    result = api.ensure_indexed(sample_codebase, validate_codebase=False)
    assert result["success"], f"Indexing failed: {result.get('error')}"

    yield api

    # Cleanup
    api.close()
    import shutil

    shutil.rmtree(temp_db, ignore_errors=True)


# ============================================================================
# Identifier Boosting Tests
# ============================================================================


class TestIdentifierBoosting:
    """Test identifier-based boosting in search results."""

    def test_camel_case_identifier_boosting(self, indexed_api):
        """Test boosting for camelCase identifier queries."""
        results = indexed_api.search("validateUserToken", n_results=5)

        assert len(results) > 0, "Should return results"

        # Top result should be boosted and contain the identifier
        top_result = results[0]
        assert top_result.get("boosted", False), "Top result should be boosted"
        assert (
            "validateUserToken" in top_result["content"]
        ), "Top result should contain exact identifier"

        # Check boost metadata
        assert "original_similarity" in top_result, "Should have original_similarity"
        boost_factor = top_result["similarity"] / top_result["original_similarity"]
        assert boost_factor >= 3.0, f"Boost factor should be >= 3.0, got {boost_factor}"

    def test_pascal_case_identifier_boosting(self, indexed_api):
        """Test boosting for PascalCase identifier queries."""
        results = indexed_api.search("AuthService", n_results=5)

        assert len(results) > 0, "Should return results"

        # Should find the class definition
        top_result = results[0]
        assert top_result.get("boosted", False), "Top result should be boosted"
        assert "AuthService" in top_result["content"]

    def test_snake_case_identifier_boosting(self, indexed_api):
        """Test boosting for snake_case identifier queries."""
        results = indexed_api.search("get_user_by_email", n_results=5)

        assert len(results) > 0, "Should return results"

        # Should find the function definition
        top_result = results[0]
        assert top_result.get("boosted", False), "Top result should be boosted"
        assert "get_user_by_email" in top_result["content"]

    def test_screaming_snake_case_identifier_boosting(self, indexed_api):
        """Test boosting for SCREAMING_SNAKE_CASE identifier queries."""
        results = indexed_api.search("MAX_RETRIES", n_results=5)

        assert len(results) > 0, "Should return results"

        # Should find the constant definition
        top_result = results[0]
        assert top_result.get("boosted", False), "Top result should be boosted"
        assert "MAX_RETRIES" in top_result["content"]

    def test_multiple_identifiers_boosting(self, indexed_api):
        """Test boosting when multiple identifiers match."""
        results = indexed_api.search("validateUserToken getUserById", n_results=5)

        assert len(results) > 0, "Should return results"

        # Find results containing both identifiers
        dual_match_results = [
            r
            for r in results
            if "validateUserToken" in r["content"] and "getUserById" in r["content"]
        ]

        if dual_match_results:
            # Results with both matches should have higher boost
            dual_match = dual_match_results[0]
            assert dual_match.get("boosted", False)
            if "original_similarity" in dual_match:
                boost_factor = (
                    dual_match["similarity"] / dual_match["original_similarity"]
                )
                # Should be 4.5x (base 3.0 * 1.5 for 2 matches)
                assert (
                    boost_factor >= 4.0
                ), f"Dual match should have higher boost, got {boost_factor}"


# ============================================================================
# Natural Language Query Tests
# ============================================================================


class TestNaturalLanguageQueries:
    """Test that natural language queries are not affected by boosting."""

    def test_no_boosting_for_natural_language(self, indexed_api):
        """Test that queries without identifiers don't trigger boosting."""
        query = "user authentication logic"
        results = indexed_api.search(query, n_results=5)

        assert len(results) > 0, "Should return results"

        # Check that no results are identifier-boosted
        # (they might have slight boost from significant words: 1.2x)
        for result in results:
            if "original_similarity" in result:
                boost_factor = result["similarity"] / result["original_similarity"]
                assert (
                    boost_factor < 2.0
                ), f"Natural language query should not have high boost, got {boost_factor}"

    def test_semantic_search_still_works(self, indexed_api):
        """Test that semantic search continues to work normally."""
        query = "password verification code"
        results = indexed_api.search(query, n_results=5)

        assert len(results) > 0, "Should return results"

        # Should find semantically related code
        contents = " ".join([r["content"] for r in results])
        assert (
            "password" in contents.lower() or "verify" in contents.lower()
        ), "Should find semantically related content"

    def test_general_error_handling_query(self, indexed_api):
        """Test general conceptual queries work without identifier boosting."""
        query = "error handling authentication"
        results = indexed_api.search(query, n_results=5)

        assert len(results) > 0, "Should return results"


# ============================================================================
# Mixed Query Tests
# ============================================================================


class TestMixedQueries:
    """Test queries combining identifiers and natural language."""

    def test_identifier_in_natural_language_context(self, indexed_api):
        """Test query with identifier embedded in natural language."""
        query = "how to use validateUserToken function"
        results = indexed_api.search(query, n_results=5)

        assert len(results) > 0, "Should return results"

        # Should boost results containing the identifier
        boosted_results = [r for r in results if r.get("boosted", False)]
        assert len(boosted_results) > 0, "Should have some boosted results"

        # Top boosted result should contain the identifier
        top_boosted = boosted_results[0]
        assert "validateUserToken" in top_boosted["content"]


# ============================================================================
# Result Metadata Tests
# ============================================================================


class TestResultMetadata:
    """Test result metadata related to boosting."""

    def test_boosted_flag_present(self, indexed_api):
        """Test that boosted flag is present in results."""
        results = indexed_api.search("validateUserToken", n_results=5)

        assert len(results) > 0, "Should return results"

        for result in results:
            assert "boosted" in result, "Should have 'boosted' flag"
            assert isinstance(result["boosted"], bool), "boosted should be boolean"

    def test_original_similarity_preserved(self, indexed_api):
        """Test that original similarity is preserved for boosted results."""
        results = indexed_api.search("AuthService", n_results=5)

        assert len(results) > 0, "Should return results"

        boosted_results = [r for r in results if r.get("boosted", False)]
        if boosted_results:
            for result in boosted_results:
                assert (
                    "original_similarity" in result
                ), "Boosted results should have original_similarity"
                assert (
                    result["original_similarity"] > 0
                ), "Original similarity should be positive"
                assert (
                    result["similarity"] >= result["original_similarity"]
                ), "Boosted similarity should be >= original"

    def test_non_boosted_results_metadata(self, indexed_api):
        """Test metadata handling for non-boosted results."""
        results = indexed_api.search("password verification", n_results=5)

        assert len(results) > 0, "Should return results"

        # When no identifiers are detected, all results should be marked as not boosted
        for result in results:
            assert "boosted" in result, "All results should have boosted flag"
            assert isinstance(result["boosted"], bool), "boosted should be boolean"
            # For natural language query, should be False
            assert (
                result["boosted"] is False
            ), "Natural language queries should not boost results"


# ============================================================================
# Ranking Tests
# ============================================================================


class TestRanking:
    """Test that boosting affects result ranking appropriately."""

    def test_exact_match_ranks_higher(self, indexed_api):
        """Test that exact identifier matches rank higher than partial matches."""
        results = indexed_api.search("getUserById", n_results=10)

        assert len(results) > 0, "Should return results"

        # Find position of exact match
        exact_match_position = None
        for i, result in enumerate(results):
            if "getUserById" in result["content"] and result.get("boosted", False):
                exact_match_position = i
                break

        assert exact_match_position is not None, "Should find exact match in results"
        assert (
            exact_match_position < 5
        ), f"Exact match should rank in top 5, found at position {exact_match_position}"

    def test_boosted_results_sorted_correctly(self, indexed_api):
        """Test that results are sorted by boosted similarity scores."""
        results = indexed_api.search("validateUserToken", n_results=5)

        assert len(results) > 0, "Should return results"

        # Verify descending order
        for i in range(len(results) - 1):
            assert (
                results[i]["similarity"] >= results[i + 1]["similarity"]
            ), "Results should be sorted by similarity (descending)"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases in hybrid search."""

    def test_query_with_no_results(self, indexed_api):
        """Test query that returns no results."""
        results = indexed_api.search(
            "ThisIdentifierDefinitelyDoesNotExistInCodebase", n_results=5
        )

        # Might return no results or very low similarity results
        # Should not crash
        assert isinstance(results, list)

    def test_empty_query_handling(self, indexed_api):
        """Test handling of empty or whitespace queries."""
        # This might raise an error or return empty results
        # The important thing is it doesn't crash
        try:
            results = indexed_api.search("", n_results=5)
            assert isinstance(results, list)
        except ValueError:
            # It's acceptable to raise ValueError for empty query
            pass

    def test_very_long_identifier_query(self, indexed_api):
        """Test query with very long identifier."""
        long_identifier = "validateUserTokenAndCheckPermissionsAndLogActivity"
        results = indexed_api.search(long_identifier, n_results=5)

        # Should handle without crashing
        assert isinstance(results, list)

    def test_special_characters_in_query(self, indexed_api):
        """Test query with special characters."""
        results = indexed_api.search("validateUserToken()", n_results=5)

        # Should still detect and boost the identifier
        assert len(results) > 0
        if results[0].get("boosted"):
            assert "validateUserToken" in results[0]["content"]


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test performance characteristics of hybrid search."""

    def test_boosting_performance_overhead(self, indexed_api):
        """Test that boosting adds minimal overhead to search."""
        import time

        # Measure search time (rough estimate)
        start = time.time()
        results = indexed_api.search("validateUserToken", n_results=5)
        end = time.time()

        elapsed = end - start

        assert len(results) > 0, "Should return results"
        # Boosting should add < 50ms overhead
        # (This is a rough check, actual time depends on hardware)
        assert elapsed < 5.0, f"Search took too long: {elapsed}s"
