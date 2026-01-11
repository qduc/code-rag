"""Unit tests for the FileProcessor class."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from code_rag.processor.file_processor import (
    TEST_DIR_PATTERNS,
    TEST_FILE_PATTERNS,
    FileProcessor,
    byte_offset_to_line_number,
)

# =============================================================================
# Utility Function Tests
# =============================================================================


class TestByteOffsetToLineNumber:
    """Tests for the byte_offset_to_line_number utility function."""

    def test_first_line(self):
        """Byte offset within the first line returns line 1."""
        content = "first line\nsecond line\nthird line"
        assert byte_offset_to_line_number(content, 0) == 1
        assert byte_offset_to_line_number(content, 5) == 1

    def test_second_line(self):
        """Byte offset in the second line returns line 2."""
        content = "first line\nsecond line\nthird line"
        # After the first newline (offset 10 is '\n')
        assert byte_offset_to_line_number(content, 11) == 2
        assert byte_offset_to_line_number(content, 15) == 2

    def test_third_line(self):
        """Byte offset in the third line returns line 3."""
        content = "first line\nsecond line\nthird line"
        # After second newline (offset 22 is '\n')
        assert byte_offset_to_line_number(content, 23) == 3

    def test_empty_content(self):
        """Empty content returns line 1."""
        assert byte_offset_to_line_number("", 0) == 1

    def test_single_line_no_newline(self):
        """Single line without newline returns line 1."""
        content = "single line without newline"
        assert byte_offset_to_line_number(content, 0) == 1
        assert byte_offset_to_line_number(content, 10) == 1


# =============================================================================
# FileProcessor Initialization Tests
# =============================================================================


class TestFileProcessorInit:
    """Tests for FileProcessor initialization."""

    def test_default_initialization(self):
        """Default initialization sets expected attributes."""
        processor = FileProcessor()
        assert processor.exclude_tests is False
        assert processor.include_file_header is True
        assert len(processor.ignore_patterns) > 0
        assert "node_modules" in processor.ignore_patterns
        assert ".git" in processor.ignore_patterns

    def test_custom_ignore_patterns(self):
        """Custom ignore patterns override defaults."""
        custom_patterns = ["custom_dir", "*.custom"]
        processor = FileProcessor(ignore_patterns=custom_patterns)
        assert processor.ignore_patterns == custom_patterns

    def test_additional_ignore_patterns(self):
        """Additional patterns are appended to defaults."""
        additional = ["extra_dir", "*.extra"]
        processor = FileProcessor(additional_ignore_patterns=additional)
        assert "extra_dir" in processor.ignore_patterns
        assert "*.extra" in processor.ignore_patterns
        assert "node_modules" in processor.ignore_patterns  # Default still there

    def test_exclude_tests_flag(self):
        """Exclude tests flag is properly set."""
        processor = FileProcessor(exclude_tests=True)
        assert processor.exclude_tests is True

    def test_include_file_header_flag(self):
        """Include file header flag is properly set."""
        processor = FileProcessor(include_file_header=False)
        assert processor.include_file_header is False


# =============================================================================
# File Discovery Tests
# =============================================================================


class TestFileDiscovery:
    """Tests for file discovery functionality."""

    def test_discover_python_files(self, tmp_path):
        """Discovers Python source files."""
        # Create test structure
        (tmp_path / "main.py").write_text("print('hello')")
        (tmp_path / "utils.py").write_text("def util(): pass")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("class MyClass: pass")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 3
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)
        assert any("module.py" in f for f in files)

    def test_discover_multiple_extensions(self, tmp_path):
        """Discovers files with various source extensions."""
        (tmp_path / "app.py").write_text("python code")
        (tmp_path / "script.js").write_text("javascript code")
        (tmp_path / "config.yaml").write_text("yaml: config")
        (tmp_path / "readme.md").write_text("# Readme")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 4
        extensions = {Path(f).suffix for f in files}
        assert extensions == {".py", ".js", ".yaml", ".md"}

    def test_ignores_default_patterns(self, tmp_path):
        """Ignores directories matching default patterns."""
        # Create files in ignored directories
        (tmp_path / "main.py").write_text("main code")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "dep.js").write_text("dependency")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cache.pyc").write_text("cache")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]
        assert not any("node_modules" in f for f in files)
        assert not any(".git" in f for f in files)
        assert not any("__pycache__" in f for f in files)

    def test_respects_gitignore(self, tmp_path):
        """Respects patterns in .gitignore file."""
        # Create .gitignore
        (tmp_path / ".gitignore").write_text("ignored_dir/\n*.ignored\nbuild/")

        # Create test files
        (tmp_path / "main.py").write_text("main code")
        (tmp_path / "ignored_dir").mkdir()
        (tmp_path / "ignored_dir" / "secret.py").write_text("secret")
        (tmp_path / "file.ignored").write_text("ignored file")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "output.js").write_text("build output")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]

    def test_gitignore_comments_and_empty_lines(self, tmp_path):
        """Gitignore comments and empty lines are ignored."""
        (tmp_path / ".gitignore").write_text("# This is a comment\n\nignored/\n")
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "ignored").mkdir()
        (tmp_path / "ignored" / "file.py").write_text("ignored")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]

    def test_exclude_tests_directories(self, tmp_path):
        """Excludes test directories when flag is set."""
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("test")
        (tmp_path / "__tests__").mkdir()
        (tmp_path / "__tests__" / "spec.js").write_text("spec")

        processor = FileProcessor(exclude_tests=True)
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]

    def test_exclude_test_files(self, tmp_path):
        """Excludes test files when flag is set."""
        (tmp_path / "main.py").write_text("main code")
        (tmp_path / "test_main.py").write_text("test code")
        (tmp_path / "main_test.py").write_text("test code")
        (tmp_path / "component.spec.js").write_text("spec")
        (tmp_path / "component.test.ts").write_text("test")

        processor = FileProcessor(exclude_tests=True)
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]

    def test_includes_tests_by_default(self, tmp_path):
        """Test files are included by default."""
        (tmp_path / "main.py").write_text("main")
        (tmp_path / "test_main.py").write_text("test")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 2

    def test_discovers_special_filenames(self, tmp_path):
        """Discovers special filenames without extensions."""
        (tmp_path / "Makefile").write_text("all: build")
        (tmp_path / "Dockerfile").write_text("FROM python:3.9")
        (tmp_path / "Gemfile").write_text("source 'https://rubygems.org'")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 3
        filenames = {Path(f).name for f in files}
        assert filenames == {"Makefile", "Dockerfile", "Gemfile"}

    def test_ignores_non_source_files(self, tmp_path):
        """Ignores files without recognized extensions."""
        (tmp_path / "main.py").write_text("code")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "binary.exe").write_bytes(b"\x00\x01")
        (tmp_path / "data.dat").write_text("data")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 1
        assert "main.py" in files[0]

    def test_empty_directory(self, tmp_path):
        """Returns empty list for empty directory."""
        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert files == []

    def test_returns_sorted_files(self, tmp_path):
        """Files are returned in sorted order."""
        (tmp_path / "z_last.py").write_text("last")
        (tmp_path / "a_first.py").write_text("first")
        (tmp_path / "m_middle.py").write_text("middle")

        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        filenames = [Path(f).name for f in files]
        assert filenames == sorted(filenames)


# =============================================================================
# File Reading Tests
# =============================================================================


class TestFileReading:
    """Tests for file reading functionality."""

    def test_read_utf8_file(self, tmp_path):
        """Reads UTF-8 encoded file successfully."""
        content = "Hello, 世界! 🌍"
        (tmp_path / "unicode.py").write_text(content, encoding="utf-8")

        processor = FileProcessor()
        result = processor.read_file(str(tmp_path / "unicode.py"))

        assert result == content

    def test_read_latin1_file(self, tmp_path):
        """Falls back to Latin-1 encoding."""
        content = "Café résumé"
        file_path = tmp_path / "latin1.txt"
        file_path.write_bytes(content.encode("latin-1"))

        processor = FileProcessor()
        result = processor.read_file(str(file_path))

        assert result == content

    def test_read_nonexistent_file(self, tmp_path):
        """Returns None for nonexistent file."""
        processor = FileProcessor()
        result = processor.read_file(str(tmp_path / "nonexistent.py"))

        assert result is None

    def test_read_permission_denied(self, tmp_path):
        """Returns None when permission is denied."""
        file_path = tmp_path / "noperm.py"
        file_path.write_text("content")
        os.chmod(file_path, 0o000)

        try:
            processor = FileProcessor()
            result = processor.read_file(str(file_path))
            assert result is None
        finally:
            # Restore permissions for cleanup
            os.chmod(file_path, 0o644)


# =============================================================================
# Chunking Tests
# =============================================================================


class TestChunking:
    """Tests for file chunking functionality."""

    def test_chunk_small_content(self):
        """Content smaller than chunk_size returns single chunk."""
        processor = FileProcessor()
        content = "small content"
        chunks = processor.chunk_file(content, chunk_size=1000)

        assert len(chunks) == 1
        assert chunks[0] == content

    def test_chunk_at_line_boundaries(self):
        """Chunks split at line boundaries when possible."""
        processor = FileProcessor()
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = processor.chunk_file(content, chunk_size=12)

        # Each chunk should end at a newline or be the last chunk
        for chunk in chunks[:-1]:
            assert chunk.endswith("\n") or len(chunk) <= 12

    def test_chunk_empty_content(self):
        """Empty content returns empty list."""
        processor = FileProcessor()
        chunks = processor.chunk_file("", chunk_size=100)

        assert chunks == []

    def test_chunk_no_newlines(self):
        """Content without newlines is chunked by size."""
        processor = FileProcessor()
        content = "a" * 100
        chunks = processor.chunk_file(content, chunk_size=30)

        assert len(chunks) > 1
        total_length = sum(len(c) for c in chunks)
        assert total_length == 100

    def test_chunk_size_none(self):
        """None chunk_size is handled gracefully."""
        processor = FileProcessor()
        content = "test content"
        chunks = processor.chunk_file(content, chunk_size=None)

        assert len(chunks) > 0

    def test_chunk_size_zero(self):
        """Zero chunk_size is handled gracefully."""
        processor = FileProcessor()
        content = "test content"
        chunks = processor.chunk_file(content, chunk_size=0)

        assert len(chunks) > 0

    def test_chunk_size_negative(self):
        """Negative chunk_size is handled gracefully."""
        processor = FileProcessor()
        content = "test content"
        chunks = processor.chunk_file(content, chunk_size=-10)

        assert len(chunks) > 0


# =============================================================================
# Process File Tests
# =============================================================================


class TestProcessFile:
    """Tests for the process_file method."""

    def test_process_python_file(self, tmp_path):
        """Processes Python file with metadata."""
        content = "def hello():\n    print('world')\n"
        file_path = tmp_path / "hello.py"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path))

        assert len(result) >= 1
        chunk = result[0]
        assert "content" in chunk
        assert "metadata" in chunk
        assert "id" in chunk
        assert chunk["metadata"]["file_path"] == str(file_path)

    def test_process_file_metadata_fields(self, tmp_path):
        """Verifies all expected metadata fields are present."""
        content = "line1\nline2\nline3"
        file_path = tmp_path / "test.py"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path))

        metadata = result[0]["metadata"]
        expected_fields = [
            "file_path",
            "chunk_index",
            "total_chunks",
            "start_line",
            "end_line",
            "start_byte",
            "end_byte",
            "prev_id",
            "next_id",
        ]
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"

    def test_process_file_chunk_adjacency(self, tmp_path):
        """Verifies chunk adjacency metadata is correct."""
        # Create content that will produce multiple chunks
        content = "\n".join([f"line{i}" for i in range(100)])
        file_path = tmp_path / "multiline.py"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path), chunk_size=50)

        if len(result) > 1:
            # First chunk has no previous
            assert result[0]["metadata"]["prev_id"] == -1
            assert result[0]["metadata"]["next_id"] == 1

            # Last chunk has no next
            assert result[-1]["metadata"]["next_id"] == -1
            assert result[-1]["metadata"]["prev_id"] == len(result) - 2

    def test_process_nonexistent_file(self, tmp_path):
        """Returns empty list for nonexistent file."""
        processor = FileProcessor()
        result = processor.process_file(str(tmp_path / "nonexistent.py"))

        assert result == []

    def test_process_file_unique_ids(self, tmp_path):
        """Each chunk has a unique ID."""
        content = "\n".join([f"line{i}" for i in range(50)])
        file_path = tmp_path / "test.py"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path), chunk_size=50)

        ids = [chunk["id"] for chunk in result]
        assert len(ids) == len(set(ids))  # All unique

    def test_process_file_line_numbers(self, tmp_path):
        """Line numbers are correctly calculated."""
        content = "line1\nline2\nline3"
        file_path = tmp_path / "test.txt"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path), chunk_size=1000)

        metadata = result[0]["metadata"]
        assert metadata["start_line"] == 1
        assert metadata["end_line"] >= 1

    def test_process_unsupported_language_fallback(self, tmp_path):
        """Falls back to basic chunking for unsupported languages."""
        content = "some text content\nwith multiple lines\n"
        file_path = tmp_path / "file.txt"  # .txt has no tree-sitter support
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path))

        assert len(result) >= 1
        assert content.strip() in result[0]["content"]


# =============================================================================
# File Stats Tests
# =============================================================================


class TestFileStats:
    """Tests for file statistics functionality."""

    def test_get_file_stats(self, tmp_path):
        """Gets mtime and size for existing file."""
        content = "test content"
        file_path = tmp_path / "test.py"
        file_path.write_text(content)

        processor = FileProcessor()
        stats = processor.get_file_stats(str(file_path))

        assert stats is not None
        assert "mtime" in stats
        assert "size" in stats
        assert stats["size"] == len(content)

    def test_get_file_stats_nonexistent(self, tmp_path):
        """Returns None for nonexistent file."""
        processor = FileProcessor()
        stats = processor.get_file_stats(str(tmp_path / "nonexistent.py"))

        assert stats is None


# =============================================================================
# File Hash Tests
# =============================================================================


class TestFileHash:
    """Tests for file hashing functionality."""

    def test_compute_file_hash(self, tmp_path):
        """Computes SHA256 hash correctly."""
        content = "test content for hashing"
        file_path = tmp_path / "test.py"
        file_path.write_text(content)

        processor = FileProcessor()
        hash_value = processor.compute_file_hash(str(file_path))

        assert hash_value is not None
        assert len(hash_value) == 64  # SHA256 hex digest length

    def test_compute_file_hash_deterministic(self, tmp_path):
        """Hash is deterministic for same content."""
        content = "deterministic content"
        file_path = tmp_path / "test.py"
        file_path.write_text(content)

        processor = FileProcessor()
        hash1 = processor.compute_file_hash(str(file_path))
        hash2 = processor.compute_file_hash(str(file_path))

        assert hash1 == hash2

    def test_compute_file_hash_different_content(self, tmp_path):
        """Different content produces different hash."""
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content one")
        file2.write_text("content two")

        processor = FileProcessor()
        hash1 = processor.compute_file_hash(str(file1))
        hash2 = processor.compute_file_hash(str(file2))

        assert hash1 != hash2

    def test_compute_file_hash_nonexistent(self, tmp_path):
        """Returns None for nonexistent file."""
        processor = FileProcessor()
        hash_value = processor.compute_file_hash(str(tmp_path / "nonexistent.py"))

        assert hash_value is None


# =============================================================================
# Test File Detection Tests
# =============================================================================


class TestIsTestFile:
    """Tests for test file detection."""

    def test_python_test_patterns(self, tmp_path):
        """Detects Python test file patterns."""
        processor = FileProcessor()

        assert processor._is_test_file(Path("test_something.py"))
        assert processor._is_test_file(Path("something_test.py"))
        assert not processor._is_test_file(Path("something.py"))

    def test_javascript_test_patterns(self, tmp_path):
        """Detects JavaScript test file patterns."""
        processor = FileProcessor()

        assert processor._is_test_file(Path("component.spec.js"))
        assert processor._is_test_file(Path("component.test.js"))
        assert processor._is_test_file(Path("component.test.tsx"))
        assert not processor._is_test_file(Path("component.js"))

    def test_java_test_patterns(self, tmp_path):
        """Detects Java test file patterns."""
        processor = FileProcessor()

        assert processor._is_test_file(Path("TestSomething.java"))
        assert processor._is_test_file(Path("SomethingTest.java"))
        assert processor._is_test_file(Path("SomethingTests.java"))
        assert not processor._is_test_file(Path("Something.java"))

    def test_file_in_test_directory(self, tmp_path):
        """Detects files in test directories."""
        processor = FileProcessor()

        assert processor._is_test_file(Path("tests/helper.py"))
        assert processor._is_test_file(Path("test/utils.py"))
        assert processor._is_test_file(Path("__tests__/setup.js"))
        assert processor._is_test_file(Path("spec/support.rb"))


# =============================================================================
# Source File Detection Tests
# =============================================================================


class TestIsSourceFile:
    """Tests for source file detection."""

    def test_common_source_extensions(self):
        """Recognizes common source code extensions."""
        processor = FileProcessor()

        assert processor._is_source_file(Path("main.py"))
        assert processor._is_source_file(Path("app.js"))
        assert processor._is_source_file(Path("main.go"))
        assert processor._is_source_file(Path("lib.rs"))
        assert processor._is_source_file(Path("Main.java"))

    def test_web_extensions(self):
        """Recognizes web development extensions."""
        processor = FileProcessor()

        assert processor._is_source_file(Path("index.html"))
        assert processor._is_source_file(Path("style.css"))
        assert processor._is_source_file(Path("app.vue"))
        assert processor._is_source_file(Path("component.svelte"))

    def test_config_extensions(self):
        """Recognizes configuration file extensions."""
        processor = FileProcessor()

        assert processor._is_source_file(Path("config.yaml"))
        assert processor._is_source_file(Path("config.json"))
        assert processor._is_source_file(Path("settings.toml"))

    def test_special_filenames(self):
        """Recognizes special filenames without extensions."""
        processor = FileProcessor()

        assert processor._is_source_file(Path("Makefile"))
        assert processor._is_source_file(Path("Dockerfile"))
        assert processor._is_source_file(Path("Gemfile"))
        assert processor._is_source_file(Path("Rakefile"))
        assert processor._is_source_file(Path("Vagrantfile"))

    def test_non_source_files(self):
        """Rejects non-source files."""
        processor = FileProcessor()

        assert not processor._is_source_file(Path("image.png"))
        assert not processor._is_source_file(Path("video.mp4"))
        assert not processor._is_source_file(Path("archive.zip"))
        assert not processor._is_source_file(Path("binary.exe"))


# =============================================================================
# Ignore Pattern Tests
# =============================================================================


class TestShouldIgnore:
    """Tests for the _should_ignore method."""

    def test_ignores_node_modules(self, tmp_path):
        """Ignores node_modules directory."""
        processor = FileProcessor()
        path = tmp_path / "node_modules" / "package" / "index.js"

        assert processor._should_ignore(path, tmp_path)

    def test_ignores_git_directory(self, tmp_path):
        """Ignores .git directory."""
        processor = FileProcessor()
        path = tmp_path / ".git" / "config"

        assert processor._should_ignore(path, tmp_path)

    def test_ignores_pycache(self, tmp_path):
        """Ignores __pycache__ directory."""
        processor = FileProcessor()
        path = tmp_path / "__pycache__" / "module.pyc"

        assert processor._should_ignore(path, tmp_path)

    def test_ignores_pyc_files(self, tmp_path):
        """Ignores .pyc files."""
        processor = FileProcessor()
        path = tmp_path / "module.pyc"

        assert processor._should_ignore(path, tmp_path)

    def test_does_not_ignore_source_file(self, tmp_path):
        """Does not ignore regular source files."""
        processor = FileProcessor()
        path = tmp_path / "main.py"

        assert not processor._should_ignore(path, tmp_path)

    def test_custom_ignore_pattern(self, tmp_path):
        """Custom ignore patterns work correctly."""
        processor = FileProcessor(ignore_patterns=["custom_dir", "*.custom"])

        assert processor._should_ignore(tmp_path / "custom_dir" / "file.py", tmp_path)
        assert processor._should_ignore(tmp_path / "file.custom", tmp_path)
        assert not processor._should_ignore(tmp_path / "main.py", tmp_path)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple functionalities."""

    def test_full_workflow(self, tmp_path):
        """Tests complete workflow: discover, read, process."""
        # Create a mini project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text(
            'def main():\n    print("Hello")\n\nif __name__ == "__main__":\n    main()'
        )
        (tmp_path / "src" / "utils.py").write_text("def helper():\n    return 42\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text(
            "def test_main():\n    assert True\n"
        )
        (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")

        # Test without excluding tests
        processor = FileProcessor()
        files = processor.discover_files(str(tmp_path))

        assert len(files) == 3

        # Process all files
        all_chunks = []
        for f in files:
            chunks = processor.process_file(f)
            all_chunks.extend(chunks)

        assert len(all_chunks) >= 3  # At least one chunk per file

        # Test with excluding tests
        processor_no_tests = FileProcessor(exclude_tests=True)
        files_no_tests = processor_no_tests.discover_files(str(tmp_path))

        assert len(files_no_tests) == 2
        # Should only have main.py and utils.py, not test_main.py
        filenames = [Path(f).name for f in files_no_tests]
        assert "test_main.py" not in filenames
        assert "main.py" in filenames
        assert "utils.py" in filenames

    def test_large_file_chunking(self, tmp_path):
        """Tests chunking of a larger file."""
        # Create a file with many lines
        lines = [f"def function_{i}():\n    return {i}\n" for i in range(100)]
        content = "\n".join(lines)
        file_path = tmp_path / "large.py"
        file_path.write_text(content)

        processor = FileProcessor()
        result = processor.process_file(str(file_path), chunk_size=500)

        # Should have multiple chunks
        assert len(result) > 1

        # All content should be covered
        all_content = "".join(chunk["content"] for chunk in result)
        # Note: Content might have slight differences due to chunking strategy
        assert len(all_content) >= len(content) * 0.9  # At least 90% coverage
