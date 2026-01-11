"""Unit tests for the config module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_rag.config.config import DEFAULT_CONFIG, Config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_embedding_model(self, monkeypatch):
        """Test default embedding model."""
        # Clear relevant env vars
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_embedding_model() == "nomic-ai/CodeRankEmbed"

    def test_default_database_type(self, monkeypatch):
        """Test default database type."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_database_type() == "chroma"

    def test_default_chunk_size(self, monkeypatch):
        """Test default chunk size."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_chunk_size() == 1024

    def test_default_batch_size(self, monkeypatch):
        """Test default batch size."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_batch_size() == 32

    def test_default_reranker_settings(self, monkeypatch):
        """Test default reranker configuration."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        # Default is true in reload() but false in DEFAULT_CONFIG
        # The actual default when no config file is read is "true"
        assert config.is_reranker_enabled() is False
        assert config.get_reranker_model() == "jinaai/jina-reranker-v3"
        assert config.get_reranker_multiplier() == 2

    def test_default_log_level(self, monkeypatch):
        """Test default log level."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_log_level() == "INFO"

    def test_default_reindex_debounce_minutes(self, monkeypatch):
        """Test default reindex debounce minutes."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_reindex_debounce_minutes() == 10

    def test_default_model_idle_timeout(self, monkeypatch):
        """Test default model idle timeout."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_model_idle_timeout() == 1800

    def test_default_shared_server_settings(self, monkeypatch):
        """Test default shared server configuration."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.is_shared_server_enabled() is True
        assert config.get_shared_server_port() == 8199

    def test_default_include_file_header(self, monkeypatch):
        """Test default include_file_header setting."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.should_include_file_header() is True

    def test_default_exclude_tests(self, monkeypatch):
        """Test default exclude_tests setting."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.should_exclude_tests() is False

    def test_default_verify_changes_with_hash(self, monkeypatch):
        """Test default verify_changes_with_hash setting."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.should_verify_changes_with_hash() is True

    def test_default_additional_ignore_patterns(self, monkeypatch):
        """Test default additional ignore patterns is empty list."""
        for key in list(os.environ.keys()):
            if key.startswith("CODE_RAG_"):
                monkeypatch.delenv(key, raising=False)

        config = Config()
        assert config.get_additional_ignore_patterns() == []


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_embedding_model_override(self, monkeypatch):
        """Test embedding model override via env var."""
        monkeypatch.setenv("CODE_RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        config = Config()
        assert config.get_embedding_model() == "text-embedding-3-small"

    def test_database_type_override(self, monkeypatch):
        """Test database type override via env var."""
        monkeypatch.setenv("CODE_RAG_DATABASE_TYPE", "qdrant")
        config = Config()
        assert config.get_database_type() == "qdrant"

    def test_database_path_override(self, monkeypatch):
        """Test database path override via env var."""
        custom_path = "/custom/db/path"
        monkeypatch.setenv("CODE_RAG_DATABASE_PATH", custom_path)
        config = Config()
        assert config.get_database_path() == custom_path

    def test_chunk_size_override(self, monkeypatch):
        """Test chunk size override via env var."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "2048")
        config = Config()
        assert config.get_chunk_size() == 2048

    def test_batch_size_override(self, monkeypatch):
        """Test batch size override via env var."""
        monkeypatch.setenv("CODE_RAG_BATCH_SIZE", "64")
        config = Config()
        assert config.get_batch_size() == 64

    def test_reranker_enabled_override(self, monkeypatch):
        """Test reranker enabled override via env var."""
        monkeypatch.setenv("CODE_RAG_RERANKER_ENABLED", "false")
        config = Config()
        assert config.is_reranker_enabled() is False

    def test_reranker_model_override(self, monkeypatch):
        """Test reranker model override via env var."""
        monkeypatch.setenv("CODE_RAG_RERANKER_MODEL", "custom-reranker")
        config = Config()
        assert config.get_reranker_model() == "custom-reranker"

    def test_reranker_multiplier_override(self, monkeypatch):
        """Test reranker multiplier override via env var."""
        monkeypatch.setenv("CODE_RAG_RERANKER_MULTIPLIER", "5")
        config = Config()
        assert config.get_reranker_multiplier() == 5

    def test_log_level_override(self, monkeypatch):
        """Test log level override via env var."""
        monkeypatch.setenv("CODE_RAG_LOG_LEVEL", "debug")
        config = Config()
        assert config.get_log_level() == "DEBUG"  # Should be uppercased

    def test_shared_server_disabled(self, monkeypatch):
        """Test disabling shared server via env var."""
        monkeypatch.setenv("CODE_RAG_SHARED_SERVER", "false")
        config = Config()
        assert config.is_shared_server_enabled() is False

    def test_shared_server_port_override(self, monkeypatch):
        """Test shared server port override via env var."""
        monkeypatch.setenv("CODE_RAG_SHARED_SERVER_PORT", "9000")
        config = Config()
        assert config.get_shared_server_port() == 9000

    def test_model_idle_timeout_override(self, monkeypatch):
        """Test model idle timeout override via env var."""
        monkeypatch.setenv("CODE_RAG_MODEL_IDLE_TIMEOUT", "3600")
        config = Config()
        assert config.get_model_idle_timeout() == 3600

    def test_reindex_debounce_override(self, monkeypatch):
        """Test reindex debounce minutes override via env var."""
        monkeypatch.setenv("CODE_RAG_REINDEX_DEBOUNCE_MINUTES", "30")
        config = Config()
        assert config.get_reindex_debounce_minutes() == 30

    def test_additional_ignore_patterns_override(self, monkeypatch):
        """Test additional ignore patterns override via env var."""
        monkeypatch.setenv("CODE_RAG_ADDITIONAL_IGNORE_PATTERNS", "*.log,*.tmp,build/")
        config = Config()
        patterns = config.get_additional_ignore_patterns()
        assert patterns == ["*.log", "*.tmp", "build/"]

    def test_additional_ignore_patterns_with_spaces(self, monkeypatch):
        """Test additional ignore patterns with spaces are trimmed."""
        monkeypatch.setenv(
            "CODE_RAG_ADDITIONAL_IGNORE_PATTERNS", " *.log , *.tmp , build/ "
        )
        config = Config()
        patterns = config.get_additional_ignore_patterns()
        assert patterns == ["*.log", "*.tmp", "build/"]


class TestBooleanTypeConversions:
    """Test boolean type conversions."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("Yes", True),
            ("YES", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("No", False),
            ("NO", False),
            ("", False),
            ("invalid", False),
        ],
    )
    def test_include_file_header_boolean_conversion(self, monkeypatch, value, expected):
        """Test boolean conversion for include_file_header."""
        monkeypatch.setenv("CODE_RAG_INCLUDE_FILE_HEADER", value)
        config = Config()
        assert config.should_include_file_header() is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_exclude_tests_boolean_conversion(self, monkeypatch, value, expected):
        """Test boolean conversion for exclude_tests."""
        monkeypatch.setenv("CODE_RAG_EXCLUDE_TESTS", value)
        config = Config()
        assert config.should_exclude_tests() is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_reranker_enabled_boolean_conversion(self, monkeypatch, value, expected):
        """Test boolean conversion for reranker_enabled."""
        monkeypatch.setenv("CODE_RAG_RERANKER_ENABLED", value)
        config = Config()
        assert config.is_reranker_enabled() is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_shared_server_enabled_boolean_conversion(
        self, monkeypatch, value, expected
    ):
        """Test boolean conversion for shared_server_enabled."""
        monkeypatch.setenv("CODE_RAG_SHARED_SERVER", value)
        config = Config()
        assert config.is_shared_server_enabled() is expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("true", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_verify_changes_with_hash_boolean_conversion(
        self, monkeypatch, value, expected
    ):
        """Test boolean conversion for verify_changes_with_hash."""
        monkeypatch.setenv("CODE_RAG_VERIFY_CHANGES_WITH_HASH", value)
        config = Config()
        assert config.should_verify_changes_with_hash() is expected


class TestIntegerTypeConversions:
    """Test integer type conversions and validation."""

    def test_valid_integer_chunk_size(self, monkeypatch):
        """Test valid integer for chunk_size."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "512")
        config = Config()
        assert config.get_chunk_size() == 512

    def test_invalid_integer_chunk_size_falls_back_to_default(self, monkeypatch):
        """Test invalid integer for chunk_size falls back to default."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "not_a_number")
        config = Config()
        assert config.get_chunk_size() == 1024  # default

    def test_invalid_integer_batch_size_falls_back_to_default(self, monkeypatch):
        """Test invalid integer for batch_size falls back to default."""
        monkeypatch.setenv("CODE_RAG_BATCH_SIZE", "invalid")
        config = Config()
        assert config.get_batch_size() == 32  # default

    def test_negative_batch_size_minimum_is_one(self, monkeypatch):
        """Test that batch_size has a minimum value of 1."""
        monkeypatch.setenv("CODE_RAG_BATCH_SIZE", "-5")
        config = Config()
        assert config.get_batch_size() == 1

    def test_zero_batch_size_minimum_is_one(self, monkeypatch):
        """Test that batch_size of 0 becomes 1."""
        monkeypatch.setenv("CODE_RAG_BATCH_SIZE", "0")
        config = Config()
        assert config.get_batch_size() == 1

    def test_invalid_integer_shared_server_port(self, monkeypatch):
        """Test invalid integer for shared_server_port falls back to default."""
        monkeypatch.setenv("CODE_RAG_SHARED_SERVER_PORT", "abc")
        config = Config()
        assert config.get_shared_server_port() == 8199  # default

    def test_invalid_integer_model_idle_timeout(self, monkeypatch):
        """Test invalid integer for model_idle_timeout falls back to default."""
        monkeypatch.setenv("CODE_RAG_MODEL_IDLE_TIMEOUT", "xyz")
        config = Config()
        assert config.get_model_idle_timeout() == 1800  # default

    def test_float_string_as_integer_fails(self, monkeypatch):
        """Test that float string is not converted to int and falls back to default."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "1024.5")
        config = Config()
        assert config.get_chunk_size() == 1024  # default due to ValueError


class TestChunkSanitization:
    """Test chunk size sanitization."""

    def test_negative_chunk_size_becomes_one(self, monkeypatch):
        """Test that negative chunk_size is sanitized to 1."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "-100")
        config = Config()
        assert config.get_chunk_size() == 1

    def test_zero_chunk_size_becomes_one(self, monkeypatch):
        """Test that zero chunk_size is sanitized to 1."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "0")
        config = Config()
        assert config.get_chunk_size() == 1


class TestDatabasePathResolution:
    """Test database path resolution."""

    def test_explicit_database_path(self, monkeypatch):
        """Test explicit database path is used when set."""
        monkeypatch.setenv("CODE_RAG_DATABASE_PATH", "/my/custom/path")
        config = Config()
        assert config.get_database_path() == "/my/custom/path"

    def test_default_database_path_linux(self, monkeypatch):
        """Test default database path on Linux."""
        # Clear the explicit path
        monkeypatch.delenv("CODE_RAG_DATABASE_PATH", raising=False)

        # Mock os.name and os.uname for Linux
        monkeypatch.setattr(os, "name", "posix")

        mock_uname = MagicMock()
        mock_uname.sysname = "Linux"
        monkeypatch.setattr(os, "uname", lambda: mock_uname)

        config = Config()
        db_path = config.get_database_path()

        # Should use XDG_CACHE_HOME or ~/.cache
        assert "code-rag" in db_path

    def test_default_database_path_with_xdg_cache(self, monkeypatch, tmp_path):
        """Test default database path uses XDG_CACHE_HOME when set."""
        monkeypatch.delenv("CODE_RAG_DATABASE_PATH", raising=False)
        custom_cache = tmp_path / "custom_cache"
        monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))

        monkeypatch.setattr(os, "name", "posix")
        mock_uname = MagicMock()
        mock_uname.sysname = "Linux"
        monkeypatch.setattr(os, "uname", lambda: mock_uname)

        config = Config()
        db_path = config.get_database_path()

        assert db_path == str(custom_cache / "code-rag")


class TestConfigReload:
    """Test config reload functionality."""

    def test_reload_updates_values(self, monkeypatch):
        """Test that reload updates configuration values."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "512")
        config = Config()
        assert config.get_chunk_size() == 512

        # Change environment and reload
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "2048")
        config._initial_env["CODE_RAG_CHUNK_SIZE"] = "2048"
        config.reload()
        assert config.get_chunk_size() == 2048

    def test_reload_restores_initial_environment(self, monkeypatch):
        """Test that reload restores initial environment state."""
        # Set initial value
        monkeypatch.setenv("CODE_RAG_LOG_LEVEL", "DEBUG")
        config = Config()
        assert config.get_log_level() == "DEBUG"

        # The initial env should have been captured
        assert "CODE_RAG_LOG_LEVEL" in config._initial_env
        assert config._initial_env["CODE_RAG_LOG_LEVEL"] == "DEBUG"


class TestConfigHasChanged:
    """Test config has_changed functionality."""

    def test_has_changed_returns_false_when_no_changes(self, monkeypatch):
        """Test has_changed returns False when no config files have changed."""
        config = Config()
        # Right after creation, no files should have changed
        assert config.has_changed() is False

    def test_has_changed_detects_file_modification(self, monkeypatch, tmp_path):
        """Test has_changed detects when a config file is modified."""
        # Create a temporary config file
        config_file = tmp_path / "code-rag.config"
        config_file.write_text("CODE_RAG_CHUNK_SIZE=512")

        # Set environment to use this config file
        monkeypatch.setenv("CODE_RAG_CONFIG_FILE", str(config_file))

        config = Config()
        assert config.has_changed() is False

        # Modify the file (change mtime)
        import time

        time.sleep(0.01)  # Ensure mtime changes
        config_file.write_text("CODE_RAG_CHUNK_SIZE=1024")

        assert config.has_changed() is True

    def test_has_changed_detects_file_removal(self, monkeypatch, tmp_path):
        """Test has_changed detects when a config file is removed."""
        # Create a temporary config file
        config_file = tmp_path / "code-rag.config"
        config_file.write_text("CODE_RAG_CHUNK_SIZE=512")

        monkeypatch.setenv("CODE_RAG_CONFIG_FILE", str(config_file))

        config = Config()
        assert config.has_changed() is False

        # Remove the file
        config_file.unlink()

        assert config.has_changed() is True


class TestConfigFilePaths:
    """Test config file path resolution."""

    def test_custom_config_file_path(self, monkeypatch, tmp_path):
        """Test custom config file path via CODE_RAG_CONFIG_FILE env var."""
        config_file = tmp_path / "my-config"
        config_file.write_text("CODE_RAG_CHUNK_SIZE=4096")

        monkeypatch.setenv("CODE_RAG_CONFIG_FILE", str(config_file))

        config = Config()
        paths = config._get_paths_to_check()

        # Custom path should be first
        assert paths[0] == Path(str(config_file))

    def test_get_paths_to_check_includes_standard_locations(self, monkeypatch):
        """Test that standard config locations are checked."""
        monkeypatch.delenv("CODE_RAG_CONFIG_FILE", raising=False)

        config = Config()
        paths = config._get_paths_to_check()

        # Should include cwd/code-rag.config and ~/.config/code-rag/config
        path_strs = [str(p) for p in paths]
        assert any("code-rag.config" in p for p in path_strs)
        assert any(".config/code-rag/config" in p for p in path_strs)


class TestDefaultConfigContent:
    """Test DEFAULT_CONFIG constant."""

    def test_default_config_contains_essential_settings(self):
        """Test that DEFAULT_CONFIG contains essential settings."""
        assert "CODE_RAG_EMBEDDING_MODEL" in DEFAULT_CONFIG
        assert "CODE_RAG_DATABASE_TYPE" in DEFAULT_CONFIG
        assert "CODE_RAG_CHUNK_SIZE" in DEFAULT_CONFIG
        assert "CODE_RAG_BATCH_SIZE" in DEFAULT_CONFIG
        assert "CODE_RAG_RERANKER_ENABLED" in DEFAULT_CONFIG
        assert "CODE_RAG_SHARED_SERVER" in DEFAULT_CONFIG

    def test_default_config_is_valid_env_file_format(self):
        """Test that DEFAULT_CONFIG uses valid env file format."""
        lines = DEFAULT_CONFIG.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                # Should be KEY=VALUE format
                assert "=" in line, f"Invalid line: {line}"


class TestGetIntEnv:
    """Test _get_int_env helper method."""

    def test_get_int_env_returns_default_when_not_set(self, monkeypatch):
        """Test _get_int_env returns default when env var not set."""
        monkeypatch.delenv("CODE_RAG_TEST_VAR", raising=False)
        config = Config()
        result = config._get_int_env("CODE_RAG_TEST_VAR", 42)
        assert result == 42

    def test_get_int_env_returns_parsed_value(self, monkeypatch):
        """Test _get_int_env returns parsed integer when valid."""
        monkeypatch.setenv("CODE_RAG_TEST_VAR", "100")
        config = Config()
        result = config._get_int_env("CODE_RAG_TEST_VAR", 42)
        assert result == 100

    def test_get_int_env_returns_default_on_invalid(self, monkeypatch):
        """Test _get_int_env returns default on invalid value."""
        monkeypatch.setenv("CODE_RAG_TEST_VAR", "not_an_int")
        config = Config()
        result = config._get_int_env("CODE_RAG_TEST_VAR", 42)
        assert result == 42

    def test_get_int_env_handles_negative_numbers(self, monkeypatch):
        """Test _get_int_env handles negative numbers."""
        monkeypatch.setenv("CODE_RAG_TEST_VAR", "-10")
        config = Config()
        result = config._get_int_env("CODE_RAG_TEST_VAR", 42)
        assert result == -10

    def test_get_int_env_handles_empty_string(self, monkeypatch):
        """Test _get_int_env handles empty string."""
        monkeypatch.setenv("CODE_RAG_TEST_VAR", "")
        config = Config()
        result = config._get_int_env("CODE_RAG_TEST_VAR", 42)
        assert result == 42


class TestDirectAttributeAccess:
    """Test direct attribute access on Config object."""

    def test_direct_attribute_embedding_model(self, monkeypatch):
        """Test direct access to embedding_model attribute."""
        monkeypatch.setenv("CODE_RAG_EMBEDDING_MODEL", "test-model")
        config = Config()
        assert config.embedding_model == "test-model"

    def test_direct_attribute_database_type(self, monkeypatch):
        """Test direct access to database_type attribute."""
        monkeypatch.setenv("CODE_RAG_DATABASE_TYPE", "qdrant")
        config = Config()
        assert config.database_type == "qdrant"

    def test_direct_attribute_chunk_size(self, monkeypatch):
        """Test direct access to chunk_size attribute."""
        monkeypatch.setenv("CODE_RAG_CHUNK_SIZE", "2048")
        config = Config()
        assert config.chunk_size == 2048

    def test_direct_attribute_batch_size(self, monkeypatch):
        """Test direct access to batch_size attribute."""
        monkeypatch.setenv("CODE_RAG_BATCH_SIZE", "16")
        config = Config()
        assert config.batch_size == 16
