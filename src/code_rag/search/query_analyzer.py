"""Query analysis utilities for detecting identifiers and improving search."""

import re
from typing import List, Set


class QueryAnalyzer:
    """Analyzes search queries to detect code identifiers and extract search hints."""

    # Patterns for different identifier styles
    CAMEL_CASE_PATTERN = re.compile(
        r"\b[a-z]+(?:[A-Z][a-z0-9]*)+\b"
    )  # validateUserToken, getUserById
    PASCAL_CASE_PATTERN = re.compile(
        r"\b[A-Z][a-z0-9]*(?:[A-Z][a-z0-9]*)+\b"
    )  # UserValidator, TokenManager
    SNAKE_CASE_PATTERN = re.compile(
        r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"
    )  # validate_user_token, get_user_by_id
    SCREAMING_SNAKE_PATTERN = re.compile(
        r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b"
    )  # MAX_RETRIES, API_KEY

    # Common words to ignore (not identifiers)
    COMMON_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "code",
        "function",
        "class",
        "method",
        "file",
        "logic",
    }

    def __init__(self, query: str):
        """
        Initialize the query analyzer.

        Args:
            query: The search query string to analyze
        """
        self.query = query
        self._identifiers: Set[str] = set()
        self._significant_words: Set[str] = set()
        self._analyze()

    def _analyze(self):
        """Analyze the query to extract identifiers and significant words."""
        # Extract identifiers matching code patterns
        self._identifiers.update(self.CAMEL_CASE_PATTERN.findall(self.query))
        self._identifiers.update(self.PASCAL_CASE_PATTERN.findall(self.query))
        self._identifiers.update(self.SNAKE_CASE_PATTERN.findall(self.query))
        self._identifiers.update(self.SCREAMING_SNAKE_PATTERN.findall(self.query))

        # Extract other significant words (not common words)
        # Split on whitespace and non-alphanumeric characters
        words = re.findall(r"\b\w+\b", self.query.lower())
        self._significant_words = {
            word for word in words if word not in self.COMMON_WORDS and len(word) > 2
        }

    @property
    def identifiers(self) -> List[str]:
        """
        Get list of detected code identifiers.

        Returns:
            List of identifier strings (camelCase, snake_case, etc.)
        """
        return list(self._identifiers)

    @property
    def significant_words(self) -> List[str]:
        """
        Get list of significant words (non-common words).

        Returns:
            List of significant word strings
        """
        return list(self._significant_words)

    def has_identifiers(self) -> bool:
        """
        Check if the query contains code identifiers.

        Returns:
            True if identifiers were detected, False otherwise
        """
        return len(self._identifiers) > 0

    def contains_exact_match(self, text: str, case_sensitive: bool = True) -> bool:
        """
        Check if the given text contains exact matches for query identifiers.

        Args:
            text: The text to search within
            case_sensitive: Whether to perform case-sensitive matching (default: True)

        Returns:
            True if any identifier is found in the text, False otherwise
        """
        if not self._identifiers:
            return False

        search_text = text if case_sensitive else text.lower()

        for identifier in self._identifiers:
            search_identifier = identifier if case_sensitive else identifier.lower()
            # Use word boundary matching to avoid partial matches
            pattern = r"\b" + re.escape(search_identifier) + r"\b"
            if re.search(pattern, search_text):
                return True

        return False

    def count_identifier_matches(self, text: str, case_sensitive: bool = True) -> int:
        """
        Count how many distinct identifiers appear in the text.

        Args:
            text: The text to search within
            case_sensitive: Whether to perform case-sensitive matching (default: True)

        Returns:
            Number of distinct identifiers found in the text
        """
        if not self._identifiers:
            return 0

        search_text = text if case_sensitive else text.lower()
        count = 0

        for identifier in self._identifiers:
            search_identifier = identifier if case_sensitive else identifier.lower()
            pattern = r"\b" + re.escape(search_identifier) + r"\b"
            if re.search(pattern, search_text):
                count += 1

        return count

    def contains_significant_words(self, text: str) -> bool:
        """
        Check if the text contains significant words from the query.

        Args:
            text: The text to search within

        Returns:
            True if any significant words are found, False otherwise
        """
        if not self._significant_words:
            return False

        text_lower = text.lower()
        return any(
            re.search(r"\b" + re.escape(word) + r"\b", text_lower)
            for word in self._significant_words
        )

    def get_boost_score(self, text: str, base_boost: float = 3.0) -> float:
        """
        Calculate a boost score for text based on identifier matches.

        Args:
            text: The text to analyze
            base_boost: Base multiplier for exact identifier match (default: 3.0)

        Returns:
            Boost multiplier (1.0 = no boost, >1.0 = boost)
        """
        if not self._identifiers:
            # No identifiers detected, use slight boost for significant words
            if self.contains_significant_words(text):
                return 1.2  # Slight boost
            return 1.0  # No boost

        # Count identifier matches
        match_count = self.count_identifier_matches(text)

        if match_count == 0:
            return 1.0  # No boost

        # Boost increases with number of matches
        # 1 match = base_boost, 2 matches = base_boost * 1.5, etc.
        return base_boost * (1 + (match_count - 1) * 0.5)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"QueryAnalyzer(query='{self.query}', "
            f"identifiers={self.identifiers}, "
            f"significant_words={self.significant_words})"
        )
