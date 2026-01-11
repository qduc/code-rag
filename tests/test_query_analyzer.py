"""Comprehensive tests for the QueryAnalyzer functionality.

These tests verify:
1. Identifier pattern detection (camelCase, PascalCase, snake_case, SCREAMING_SNAKE_CASE)
2. Exact match detection in code chunks
3. Boost score calculation
4. Significant word extraction
5. Edge cases and boundary conditions
"""

from code_rag.search.query_analyzer import QueryAnalyzer

# ============================================================================
# Pattern Detection Tests
# ============================================================================


class TestPatternDetection:
    """Test identifier pattern detection capabilities."""

    def test_camel_case_detection(self):
        """Test detection of camelCase identifiers."""
        test_cases = [
            ("validateUserToken", ["validateUserToken"]),
            ("getUserById", ["getUserById"]),
            ("processPayment", ["processPayment"]),
            ("handleRequest", ["handleRequest"]),
        ]

        for query, expected_identifiers in test_cases:
            analyzer = QueryAnalyzer(query)
            assert analyzer.has_identifiers(), f"Should detect identifier in '{query}'"
            assert set(analyzer.identifiers) == set(
                expected_identifiers
            ), f"Expected {expected_identifiers} but got {analyzer.identifiers}"

    def test_pascal_case_detection(self):
        """Test detection of PascalCase identifiers."""
        test_cases = [
            ("AuthService", ["AuthService"]),
            ("UserValidator", ["UserValidator"]),
            ("TokenManager", ["TokenManager"]),
            ("DatabaseConnection", ["DatabaseConnection"]),
        ]

        for query, expected_identifiers in test_cases:
            analyzer = QueryAnalyzer(query)
            assert analyzer.has_identifiers(), f"Should detect identifier in '{query}'"
            assert set(analyzer.identifiers) == set(expected_identifiers)

    def test_snake_case_detection(self):
        """Test detection of snake_case identifiers."""
        test_cases = [
            ("validate_user_token", ["validate_user_token"]),
            ("get_user_by_id", ["get_user_by_id"]),
            ("process_payment", ["process_payment"]),
            ("create_database_connection", ["create_database_connection"]),
        ]

        for query, expected_identifiers in test_cases:
            analyzer = QueryAnalyzer(query)
            assert analyzer.has_identifiers(), f"Should detect identifier in '{query}'"
            assert set(analyzer.identifiers) == set(expected_identifiers)

    def test_screaming_snake_case_detection(self):
        """Test detection of SCREAMING_SNAKE_CASE identifiers."""
        test_cases = [
            ("MAX_RETRIES", ["MAX_RETRIES"]),
            ("API_KEY", ["API_KEY"]),
            ("DEFAULT_TIMEOUT", ["DEFAULT_TIMEOUT"]),
            ("DATABASE_URL", ["DATABASE_URL"]),
        ]

        for query, expected_identifiers in test_cases:
            analyzer = QueryAnalyzer(query)
            assert analyzer.has_identifiers(), f"Should detect identifier in '{query}'"
            assert set(analyzer.identifiers) == set(expected_identifiers)

    def test_multiple_identifiers_in_query(self):
        """Test detection of multiple identifiers in a single query."""
        query = "validateUserToken getUserById AuthService"
        analyzer = QueryAnalyzer(query)

        assert analyzer.has_identifiers()
        expected = {"validateUserToken", "getUserById", "AuthService"}
        assert set(analyzer.identifiers) == expected

    def test_mixed_case_styles_in_query(self):
        """Test detection of mixed identifier styles in one query."""
        query = "validateToken process_payment MAX_RETRIES AuthService"
        analyzer = QueryAnalyzer(query)

        assert analyzer.has_identifiers()
        expected = {"validateToken", "process_payment", "MAX_RETRIES", "AuthService"}
        assert set(analyzer.identifiers) == expected

    def test_no_identifiers_in_natural_language(self):
        """Test that natural language queries don't trigger identifier detection."""
        test_cases = [
            "how does authentication work",
            "code that handles errors",
            "retry logic when upstream fails",
            "user authentication flow",
        ]

        for query in test_cases:
            analyzer = QueryAnalyzer(query)
            assert (
                not analyzer.has_identifiers()
            ), f"Should not detect identifiers in '{query}'"


# ============================================================================
# Exact Match Detection Tests
# ============================================================================


class TestExactMatchDetection:
    """Test exact match detection in code chunks."""

    def test_exact_match_in_function_definition(self):
        """Test exact match detection in function definitions."""
        query = "validateUserToken"
        code = "def validateUserToken(token):\n    return token.is_valid()"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_exact_match(code)

    def test_exact_match_in_class_definition(self):
        """Test exact match detection in class definitions."""
        query = "AuthService"
        code = "class AuthService:\n    def __init__(self):\n        pass"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_exact_match(code)

    def test_exact_match_in_method_call(self):
        """Test exact match detection in method calls."""
        query = "getUserById"
        code = "user = getUserById(user_id)"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_exact_match(code)

    def test_no_match_for_partial_identifier(self):
        """Test that partial matches don't count as exact matches."""
        query = "validateUserToken"
        code = "# Validate user token before proceeding"

        analyzer = QueryAnalyzer(query)
        assert not analyzer.contains_exact_match(code)

    def test_no_match_for_different_identifier(self):
        """Test that different identifiers don't match."""
        query = "validateUserToken"
        code = "class TokenValidator:\n    pass"

        analyzer = QueryAnalyzer(query)
        assert not analyzer.contains_exact_match(code)

    def test_case_sensitive_matching(self):
        """Test that matching is case-sensitive by default."""
        query = "validateToken"
        code = "def ValidateToken():\n    pass"

        analyzer = QueryAnalyzer(query)
        assert not analyzer.contains_exact_match(
            code, case_sensitive=True
        ), "Should not match different case"

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching when enabled."""
        query = "validateToken"
        code = "def ValidateToken():\n    pass"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_exact_match(
            code, case_sensitive=False
        ), "Should match ignoring case"

    def test_word_boundary_matching(self):
        """Test that matches respect word boundaries."""
        query = "Token"
        code = "class UserToken:\n    pass"

        analyzer = QueryAnalyzer(query)
        # "Token" should not match as part of "UserToken"
        assert not analyzer.contains_exact_match(code)

    def test_multiple_identifier_matches(self):
        """Test counting multiple identifier matches in code."""
        query = "validateUserToken getUserById"
        code = "def validateUserToken(token):\n    user = getUserById(token.user_id)"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_exact_match(code)
        assert analyzer.count_identifier_matches(code) == 2


# ============================================================================
# Boost Score Calculation Tests
# ============================================================================


class TestBoostScoreCalculation:
    """Test boost score calculation logic."""

    def test_no_boost_for_no_identifiers(self):
        """Test that queries without identifiers get minimal or no boost."""
        query = "authentication logic"
        code = "def authenticate(user):\n    pass"

        analyzer = QueryAnalyzer(query)
        boost = analyzer.get_boost_score(code)
        # Non-identifier queries should have minimal or no boost
        assert (
            boost < 2.0
        ), f"Boost should be < 2.0 for non-identifier queries, got {boost}"

    def test_slight_boost_for_significant_words(self):
        """Test that significant words provide slight boost."""
        query = "authentication logic"
        code = "This code handles authentication and business logic"

        analyzer = QueryAnalyzer(query)
        boost = analyzer.get_boost_score(code)
        # Should get slight boost for significant word matches
        assert boost >= 1.0 and boost < 2.0, f"Slight boost expected, got {boost}"

    def test_no_boost_for_no_match(self):
        """Test that code without matches gets no boost."""
        query = "validateToken"
        code = "def processPayment():\n    pass"

        analyzer = QueryAnalyzer(query)
        boost = analyzer.get_boost_score(code)
        assert boost == 1.0

    def test_base_boost_for_single_match(self):
        """Test base boost for single identifier match."""
        query = "validateToken"
        code = "def validateToken(token):\n    pass"

        analyzer = QueryAnalyzer(query)
        boost = analyzer.get_boost_score(code)
        # Single identifier match should have significant boost (> 2.0)
        assert boost >= 2.0, f"Single match should have boost >= 2.0, got {boost}"

    def test_increased_boost_for_multiple_matches(self):
        """Test increased boost for multiple identifier matches."""
        query = "validateToken getUserById"
        code = "def validateToken(token):\n    user = getUserById(token.user_id)"

        # Get single match boost for comparison
        single_analyzer = QueryAnalyzer("validateToken")
        single_boost = single_analyzer.get_boost_score(
            "def validateToken(token):\n    pass"
        )

        analyzer = QueryAnalyzer(query)
        boost = analyzer.get_boost_score(code)
        # Multiple matches should have higher boost than single match
        assert (
            boost > single_boost
        ), f"Multiple matches ({boost}) should boost more than single ({single_boost})"

    def test_custom_base_boost(self):
        """Test using custom base_boost parameter."""
        query = "validateToken"
        code = "def validateToken(token):\n    pass"

        analyzer = QueryAnalyzer(query)
        default_boost = analyzer.get_boost_score(code)
        custom_boost = analyzer.get_boost_score(code, base_boost=5.0)
        # Custom base_boost should affect the result
        assert custom_boost >= 5.0, f"Custom boost should be >= 5.0, got {custom_boost}"
        assert custom_boost != default_boost, "Custom boost should differ from default"

    def test_boost_scaling_with_match_count(self):
        """Test that boost scales with match count."""
        single_query = "validateToken"
        single_code = "def validateToken(token):\n    pass"

        double_query = "validateToken getUserById"
        double_code = "def validateToken(token):\n    user = getUserById(token.user_id)"

        triple_query = "validateToken getUserById processPayment"
        triple_code = "def validateToken(token):\n    user = getUserById(token.user_id)\n    processPayment(user)"

        single_boost = QueryAnalyzer(single_query).get_boost_score(single_code)
        double_boost = QueryAnalyzer(double_query).get_boost_score(double_code)
        triple_boost = QueryAnalyzer(triple_query).get_boost_score(triple_code)

        # Boost should increase with more matches
        assert (
            triple_boost > double_boost > single_boost
        ), f"Boost should scale: {single_boost} < {double_boost} < {triple_boost}"


# ============================================================================
# Significant Words Tests
# ============================================================================


class TestSignificantWords:
    """Test significant word extraction."""

    def test_extracts_significant_words(self):
        """Test extraction of significant words from query."""
        query = "authentication token validation"
        analyzer = QueryAnalyzer(query)

        significant = set(analyzer.significant_words)
        assert "authentication" in significant
        assert "token" in significant
        assert "validation" in significant

    def test_filters_common_words(self):
        """Test that common words are filtered out."""
        query = "the authentication and validation logic"
        analyzer = QueryAnalyzer(query)

        significant = set(analyzer.significant_words)
        assert "the" not in significant
        assert "and" not in significant
        assert "authentication" in significant
        assert "validation" in significant

    def test_filters_short_words(self):
        """Test that words shorter than 3 characters are filtered."""
        query = "to be or not to be"
        analyzer = QueryAnalyzer(query)

        significant = set(analyzer.significant_words)
        assert "to" not in significant
        assert "be" not in significant
        assert "or" not in significant
        assert "not" in significant  # 3 characters, should be included

    def test_contains_significant_words_detection(self):
        """Test detection of significant words in text."""
        query = "authentication token"
        code = "This code handles authentication and token validation"

        analyzer = QueryAnalyzer(query)
        assert analyzer.contains_significant_words(code)


# ============================================================================
# Edge Cases and Boundary Conditions
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self):
        """Test handling of empty query."""
        analyzer = QueryAnalyzer("")

        assert not analyzer.has_identifiers()
        assert len(analyzer.identifiers) == 0
        assert len(analyzer.significant_words) == 0
        assert analyzer.get_boost_score("any code") == 1.0

    def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        analyzer = QueryAnalyzer("   \n\t  ")

        assert not analyzer.has_identifiers()
        assert len(analyzer.identifiers) == 0

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        query = "validateToken() { return true; }"
        analyzer = QueryAnalyzer(query)

        # Should still detect validateToken
        assert "validateToken" in analyzer.identifiers

    def test_numbers_in_identifiers(self):
        """Test identifiers containing numbers."""
        query = "getUserById2 processPayment3"
        analyzer = QueryAnalyzer(query)

        assert "getUserById2" in analyzer.identifiers
        assert "processPayment3" in analyzer.identifiers

    def test_single_letter_identifier_not_detected(self):
        """Test that single letters are not detected as identifiers."""
        query = "a b c x y z"
        analyzer = QueryAnalyzer(query)

        assert not analyzer.has_identifiers()

    def test_repr_method(self):
        """Test string representation of QueryAnalyzer."""
        query = "validateToken"
        analyzer = QueryAnalyzer(query)

        repr_str = repr(analyzer)
        assert "QueryAnalyzer" in repr_str
        assert "validateToken" in repr_str

    def test_unicode_handling(self):
        """Test handling of unicode characters in query."""
        query = "validateToken função"
        analyzer = QueryAnalyzer(query)

        # Should detect ASCII identifier
        assert "validateToken" in analyzer.identifiers

    def test_very_long_identifier(self):
        """Test handling of very long identifiers."""
        long_identifier = "validateUserTokenAndCheckPermissionsAndLogActivity"
        query = long_identifier
        analyzer = QueryAnalyzer(query)

        assert long_identifier in analyzer.identifiers


# ============================================================================
# Integration Tests
# ============================================================================


class TestQueryAnalyzerIntegration:
    """Integration tests combining multiple features."""

    def test_complex_query_with_mixed_content(self):
        """Test complex query with identifiers and natural language."""
        query = "how to use validateUserToken and processPayment methods"
        analyzer = QueryAnalyzer(query)

        # Should detect identifiers
        assert "validateUserToken" in analyzer.identifiers
        assert "processPayment" in analyzer.identifiers

        # Should extract significant words
        assert "methods" in analyzer.significant_words

    def test_realistic_code_search_scenario(self):
        """Test realistic code search scenario."""
        query = "getUserById"
        code_snippet = """
class UserService:
    def getUserById(self, user_id):
        '''Retrieve user by ID from database'''
        return self.db.query(User).filter(User.id == user_id).first()
"""

        analyzer = QueryAnalyzer(query)

        # Should match
        assert analyzer.contains_exact_match(code_snippet)

        # Should boost (exact identifier match)
        boost = analyzer.get_boost_score(code_snippet)
        assert boost > 1.0, f"Identifier match should have boost > 1.0, got {boost}"

    def test_no_false_positives_on_comments(self):
        """Test that commented-out identifiers still count as matches."""
        query = "validateToken"
        code = "# TODO: implement validateToken function"

        analyzer = QueryAnalyzer(query)

        # Should still match (word boundary matching)
        assert analyzer.contains_exact_match(code)
