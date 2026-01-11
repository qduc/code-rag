"""Comprehensive tests for the MetadataIndex functionality.

These tests verify:
1. Index creation and loading from disk
2. Adding and updating file metadata (mtime, size, sha256)
3. Checking if files need reindexing (needs_reindex)
4. Getting files that need reindexing (get_files_needing_reindex)
5. Removing file entries
6. Saving and loading index state
7. Handling of corrupted/missing index files
8. Hash verification logic
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_rag.index.metadata_index import FileMetadata, MetadataIndex

# ============================================================================
# FileMetadata Tests
# ============================================================================


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_file_metadata_creation(self):
        """Test creating a FileMetadata instance."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
            content_hash="abc123",
            last_indexed=1234567890.0,
        )

        assert metadata.file_path == "/path/to/file.py"
        assert metadata.mtime == 1234567890.0
        assert metadata.size == 1024
        assert metadata.chunk_count == 5
        assert metadata.content_hash == "abc123"
        assert metadata.last_indexed == 1234567890.0

    def test_file_metadata_default_values(self):
        """Test FileMetadata with default values."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=3,
        )

        assert metadata.content_hash is None
        assert metadata.last_indexed == 0.0

    def test_file_metadata_to_dict(self):
        """Test converting FileMetadata to dictionary."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
            content_hash="abc123",
            last_indexed=1234567890.0,
        )

        result = metadata.to_dict()

        assert result == {
            "file_path": "/path/to/file.py",
            "mtime": 1234567890.0,
            "size": 1024,
            "chunk_count": 5,
            "content_hash": "abc123",
            "last_indexed": 1234567890.0,
        }

    def test_file_metadata_from_dict(self):
        """Test creating FileMetadata from dictionary."""
        data = {
            "file_path": "/path/to/file.py",
            "mtime": 1234567890.0,
            "size": 1024,
            "chunk_count": 5,
            "content_hash": "abc123",
            "last_indexed": 1234567890.0,
        }

        metadata = FileMetadata.from_dict(data)

        assert metadata.file_path == "/path/to/file.py"
        assert metadata.mtime == 1234567890.0
        assert metadata.size == 1024
        assert metadata.chunk_count == 5
        assert metadata.content_hash == "abc123"
        assert metadata.last_indexed == 1234567890.0

    def test_file_metadata_round_trip(self):
        """Test that to_dict and from_dict are inverses."""
        original = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
            content_hash="abc123",
            last_indexed=1234567890.0,
        )

        restored = FileMetadata.from_dict(original.to_dict())

        assert restored.file_path == original.file_path
        assert restored.mtime == original.mtime
        assert restored.size == original.size
        assert restored.chunk_count == original.chunk_count
        assert restored.content_hash == original.content_hash
        assert restored.last_indexed == original.last_indexed


# ============================================================================
# MetadataIndex Initialization Tests
# ============================================================================


class TestMetadataIndexInit:
    """Tests for MetadataIndex initialization."""

    def test_init_creates_empty_index(self, tmp_path):
        """Test that initialization creates an empty index when file doesn't exist."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        assert index.metadata == {}
        assert index.last_reindex_time == 0.0

    def test_init_loads_existing_index(self, tmp_path):
        """Test that initialization loads an existing index file."""
        index_path = tmp_path / "metadata.json"

        # Create an existing index file
        data = {
            "last_reindex_time": 1234567890.0,
            "files": {
                "/path/to/file.py": {
                    "file_path": "/path/to/file.py",
                    "mtime": 1234567890.0,
                    "size": 1024,
                    "chunk_count": 5,
                    "content_hash": "abc123",
                    "last_indexed": 1234567890.0,
                }
            },
        }
        with open(index_path, "w") as f:
            json.dump(data, f)

        index = MetadataIndex(str(index_path))

        assert index.last_reindex_time == 1234567890.0
        assert len(index.metadata) == 1
        assert "/path/to/file.py" in index.metadata

    def test_init_handles_corrupted_json(self, tmp_path, capsys):
        """Test that initialization handles corrupted JSON gracefully."""
        index_path = tmp_path / "metadata.json"

        # Create a corrupted index file
        with open(index_path, "w") as f:
            f.write("{ invalid json }")

        index = MetadataIndex(str(index_path))

        assert index.metadata == {}
        assert index.last_reindex_time == 0.0

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Corrupted metadata index" in captured.out

    def test_init_handles_missing_files_key(self, tmp_path):
        """Test initialization with missing 'files' key in JSON."""
        index_path = tmp_path / "metadata.json"

        data = {"last_reindex_time": 1234567890.0}
        with open(index_path, "w") as f:
            json.dump(data, f)

        index = MetadataIndex(str(index_path))

        assert index.last_reindex_time == 1234567890.0
        assert index.metadata == {}

    def test_init_handles_missing_last_reindex_time(self, tmp_path):
        """Test initialization with missing 'last_reindex_time' key."""
        index_path = tmp_path / "metadata.json"

        data = {
            "files": {
                "/path/to/file.py": {
                    "file_path": "/path/to/file.py",
                    "mtime": 1234567890.0,
                    "size": 1024,
                    "chunk_count": 5,
                }
            }
        }
        with open(index_path, "w") as f:
            json.dump(data, f)

        index = MetadataIndex(str(index_path))

        assert index.last_reindex_time == 0.0
        assert len(index.metadata) == 1


# ============================================================================
# MetadataIndex Save/Load Tests
# ============================================================================


class TestMetadataIndexSaveLoad:
    """Tests for saving and loading MetadataIndex."""

    def test_save_creates_directory_if_needed(self, tmp_path):
        """Test that save creates parent directories if they don't exist."""
        index_path = tmp_path / "nested" / "dir" / "metadata.json"
        index = MetadataIndex(str(index_path))

        index.metadata["/path/to/file.py"] = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        index.save()

        assert index_path.exists()

    def test_save_writes_correct_format(self, tmp_path):
        """Test that save writes the correct JSON format."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        index.metadata["/path/to/file.py"] = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
            content_hash="abc123",
            last_indexed=1234567890.0,
        )
        index.last_reindex_time = 9999999999.0
        index.save()

        with open(index_path, "r") as f:
            data = json.load(f)

        assert data["last_reindex_time"] == 9999999999.0
        assert "/path/to/file.py" in data["files"]
        assert data["files"]["/path/to/file.py"]["size"] == 1024

    def test_save_atomic_write(self, tmp_path):
        """Test that save uses atomic write (temp file replacement)."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        index.metadata["/path/to/file.py"] = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        index.save()

        # Temp file should not exist after save
        temp_path = index_path.with_suffix(".tmp")
        assert not temp_path.exists()
        assert index_path.exists()

    def test_round_trip_save_load(self, tmp_path):
        """Test that save and load produce identical state."""
        index_path = tmp_path / "metadata.json"

        # Create and save
        index1 = MetadataIndex(str(index_path))
        index1.metadata["/path/to/file.py"] = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
            content_hash="abc123",
            last_indexed=1234567890.0,
        )
        index1.last_reindex_time = 9999999999.0
        index1.save()

        # Load into new instance
        index2 = MetadataIndex(str(index_path))

        assert index2.last_reindex_time == index1.last_reindex_time
        assert len(index2.metadata) == len(index1.metadata)
        assert index2.metadata["/path/to/file.py"].size == 1024


# ============================================================================
# File Metadata CRUD Tests
# ============================================================================


class TestMetadataIndexCRUD:
    """Tests for CRUD operations on file metadata."""

    @pytest.fixture
    def index(self, tmp_path):
        """Provide a fresh MetadataIndex instance."""
        index_path = tmp_path / "metadata.json"
        return MetadataIndex(str(index_path))

    def test_get_file_metadata_returns_none_for_unknown(self, index):
        """Test that get_file_metadata returns None for unknown files."""
        result = index.get_file_metadata("/unknown/file.py")
        assert result is None

    def test_get_file_metadata_returns_metadata(self, index):
        """Test that get_file_metadata returns existing metadata."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        index.metadata["/path/to/file.py"] = metadata

        result = index.get_file_metadata("/path/to/file.py")

        assert result is not None
        assert result.size == 1024

    def test_update_file_metadata_adds_new(self, index):
        """Test that update_file_metadata adds a new entry."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )

        index.update_file_metadata("/path/to/file.py", metadata)

        assert "/path/to/file.py" in index.metadata
        assert index.metadata["/path/to/file.py"].size == 1024

    def test_update_file_metadata_overwrites_existing(self, index):
        """Test that update_file_metadata overwrites existing entry."""
        metadata1 = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        metadata2 = FileMetadata(
            file_path="/path/to/file.py",
            mtime=9999999999.0,
            size=2048,
            chunk_count=10,
        )

        index.update_file_metadata("/path/to/file.py", metadata1)
        index.update_file_metadata("/path/to/file.py", metadata2)

        assert index.metadata["/path/to/file.py"].size == 2048
        assert index.metadata["/path/to/file.py"].chunk_count == 10

    def test_remove_file_metadata_removes_existing(self, index):
        """Test that remove_file_metadata removes an existing entry."""
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        index.metadata["/path/to/file.py"] = metadata

        index.remove_file_metadata("/path/to/file.py")

        assert "/path/to/file.py" not in index.metadata

    def test_remove_file_metadata_handles_unknown_file(self, index):
        """Test that remove_file_metadata handles unknown files gracefully."""
        # Should not raise an exception
        index.remove_file_metadata("/unknown/file.py")
        assert "/unknown/file.py" not in index.metadata

    def test_get_all_tracked_files(self, index):
        """Test that get_all_tracked_files returns all tracked file paths."""
        files = ["/path/to/file1.py", "/path/to/file2.py", "/path/to/file3.py"]
        for file_path in files:
            index.metadata[file_path] = FileMetadata(
                file_path=file_path,
                mtime=1234567890.0,
                size=1024,
                chunk_count=5,
            )

        result = index.get_all_tracked_files()

        assert result == set(files)

    def test_get_all_tracked_files_empty(self, index):
        """Test that get_all_tracked_files returns empty set when empty."""
        result = index.get_all_tracked_files()
        assert result == set()


# ============================================================================
# Reindex Timing Tests
# ============================================================================


class TestReindexTiming:
    """Tests for reindex timing functionality."""

    @pytest.fixture
    def index(self, tmp_path):
        """Provide a fresh MetadataIndex instance."""
        index_path = tmp_path / "metadata.json"
        return MetadataIndex(str(index_path))

    def test_should_reindex_returns_true_when_never_indexed(self, index):
        """Test that should_reindex returns True when never indexed."""
        assert index.last_reindex_time == 0.0
        assert index.should_reindex(debounce_minutes=60) is True

    def test_should_reindex_returns_true_after_debounce_period(self, index):
        """Test that should_reindex returns True after debounce period."""
        # Set last reindex time to 2 hours ago
        index.last_reindex_time = datetime.now().timestamp() - 7200

        assert index.should_reindex(debounce_minutes=60) is True

    def test_should_reindex_returns_false_within_debounce_period(self, index):
        """Test that should_reindex returns False within debounce period."""
        # Set last reindex time to now
        index.last_reindex_time = datetime.now().timestamp()

        assert index.should_reindex(debounce_minutes=60) is False

    def test_should_reindex_boundary_condition(self, index):
        """Test should_reindex at the exact boundary."""
        # Set last reindex time to exactly debounce_minutes ago
        debounce_minutes = 30
        index.last_reindex_time = datetime.now().timestamp() - (debounce_minutes * 60)

        # At the exact boundary, should return True (>= check)
        assert index.should_reindex(debounce_minutes=debounce_minutes) is True

    def test_mark_reindex_complete_updates_time(self, index, tmp_path):
        """Test that mark_reindex_complete updates the timestamp."""
        assert index.last_reindex_time == 0.0

        before = datetime.now().timestamp()
        index.mark_reindex_complete()
        after = datetime.now().timestamp()

        assert before <= index.last_reindex_time <= after

    def test_mark_reindex_complete_saves_to_disk(self, tmp_path):
        """Test that mark_reindex_complete persists to disk."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        index.mark_reindex_complete()

        # Reload and verify
        index2 = MetadataIndex(str(index_path))
        assert index2.last_reindex_time > 0.0


# ============================================================================
# Hash Computation Tests
# ============================================================================


class TestHashComputation:
    """Tests for file hash computation."""

    @pytest.fixture
    def index(self, tmp_path):
        """Provide a fresh MetadataIndex instance."""
        index_path = tmp_path / "metadata.json"
        return MetadataIndex(str(index_path))

    def test_compute_file_hash_returns_sha256(self, index, tmp_path):
        """Test that compute_file_hash returns correct SHA256 hash."""
        test_file = tmp_path / "test.py"
        content = b"print('hello world')"
        test_file.write_bytes(content)

        result = index.compute_file_hash(str(test_file))

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_compute_file_hash_handles_large_files(self, index, tmp_path):
        """Test that compute_file_hash works with large files."""
        test_file = tmp_path / "large_file.bin"
        # Create a file larger than the 8KB chunk size
        content = b"x" * 100000
        test_file.write_bytes(content)

        result = index.compute_file_hash(str(test_file))

        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_compute_file_hash_returns_none_for_missing_file(self, index, capsys):
        """Test that compute_file_hash returns None for missing files."""
        result = index.compute_file_hash("/nonexistent/file.py")

        assert result is None
        captured = capsys.readouterr()
        assert "Error computing hash" in captured.out

    def test_compute_file_hash_handles_permission_error(self, index, tmp_path, capsys):
        """Test that compute_file_hash handles permission errors."""
        test_file = tmp_path / "protected.py"
        test_file.write_text("content")
        test_file.chmod(0o000)

        try:
            result = index.compute_file_hash(str(test_file))

            assert result is None
            captured = capsys.readouterr()
            assert "Error computing hash" in captured.out
        finally:
            # Restore permissions for cleanup
            test_file.chmod(0o644)

    def test_compute_file_hash_empty_file(self, index, tmp_path):
        """Test compute_file_hash with an empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_bytes(b"")

        result = index.compute_file_hash(str(test_file))

        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


# ============================================================================
# Change Detection Tests
# ============================================================================


class TestChangeDetection:
    """Tests for detect_changes functionality."""

    @pytest.fixture
    def index_with_files(self, tmp_path):
        """Provide a MetadataIndex with some existing files tracked."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Create real test files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content1")
        file2.write_text("content2")

        stat1 = file1.stat()
        stat2 = file2.stat()

        # Track them in the index
        index.metadata[str(file1)] = FileMetadata(
            file_path=str(file1),
            mtime=stat1.st_mtime,
            size=stat1.st_size,
            chunk_count=1,
            content_hash=index.compute_file_hash(str(file1)),
        )
        index.metadata[str(file2)] = FileMetadata(
            file_path=str(file2),
            mtime=stat2.st_mtime,
            size=stat2.st_size,
            chunk_count=1,
            content_hash=index.compute_file_hash(str(file2)),
        )

        return index, tmp_path

    def test_detect_changes_identifies_new_files(self, index_with_files):
        """Test that detect_changes identifies new files."""
        index, tmp_path = index_with_files

        # Create a new file
        new_file = tmp_path / "new_file.py"
        new_file.write_text("new content")

        current_files = [
            str(tmp_path / "file1.py"),
            str(tmp_path / "file2.py"),
            str(new_file),
        ]
        result = index.detect_changes(current_files)

        assert str(new_file) in result["added"]
        assert len(result["modified"]) == 0
        assert len(result["deleted"]) == 0

    def test_detect_changes_identifies_deleted_files(self, index_with_files):
        """Test that detect_changes identifies deleted files."""
        index, tmp_path = index_with_files

        # Only pass one of the tracked files
        current_files = [str(tmp_path / "file1.py")]
        result = index.detect_changes(current_files)

        assert str(tmp_path / "file2.py") in result["deleted"]
        assert len(result["added"]) == 0
        assert len(result["modified"]) == 0

    def test_detect_changes_identifies_modified_files_mtime(self, index_with_files):
        """Test that detect_changes identifies modified files by mtime."""
        index, tmp_path = index_with_files

        file1 = tmp_path / "file1.py"
        # Modify the file content
        time.sleep(0.01)  # Ensure mtime changes
        file1.write_text("modified content")

        current_files = [str(file1), str(tmp_path / "file2.py")]
        result = index.detect_changes(current_files, verify_with_hash=False)

        assert str(file1) in result["modified"]

    def test_detect_changes_with_hash_verification(self, index_with_files):
        """Test detect_changes with hash verification enabled."""
        index, tmp_path = index_with_files

        file1 = tmp_path / "file1.py"
        # Modify the file content
        time.sleep(0.01)
        file1.write_text("modified content for hash test")

        current_files = [str(file1), str(tmp_path / "file2.py")]
        result = index.detect_changes(current_files, verify_with_hash=True)

        assert str(file1) in result["modified"]

    def test_detect_changes_false_positive_with_hash_verification(
        self, index_with_files
    ):
        """Test that hash verification catches false positives."""
        index, tmp_path = index_with_files

        file1 = tmp_path / "file1.py"
        original_content = file1.read_text()

        # Touch the file (change mtime) but keep same content
        time.sleep(0.01)
        file1.write_text(original_content)

        current_files = [str(file1), str(tmp_path / "file2.py")]
        result = index.detect_changes(current_files, verify_with_hash=True)

        # File should be marked as unchanged because hash matches
        assert str(file1) in result["unchanged"]
        assert str(file1) not in result["modified"]

    def test_detect_changes_unchanged_files(self, index_with_files):
        """Test that unchanged files are correctly identified."""
        index, tmp_path = index_with_files

        current_files = [str(tmp_path / "file1.py"), str(tmp_path / "file2.py")]
        result = index.detect_changes(current_files)

        assert len(result["unchanged"]) == 2
        assert len(result["added"]) == 0
        assert len(result["modified"]) == 0
        assert len(result["deleted"]) == 0

    def test_detect_changes_handles_file_access_error(self, tmp_path, capsys):
        """Test that detect_changes handles file access errors gracefully."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Pass a non-existent file as current
        current_files = ["/nonexistent/file.py"]
        result = index.detect_changes(current_files)

        # File that can't be stat'd is treated as deleted
        assert "/nonexistent/file.py" in result["deleted"]

    def test_detect_changes_empty_current_files(self, index_with_files):
        """Test detect_changes when all files are deleted."""
        index, tmp_path = index_with_files

        result = index.detect_changes([])

        assert len(result["deleted"]) == 2
        assert len(result["added"]) == 0
        assert len(result["modified"]) == 0
        assert len(result["unchanged"]) == 0

    def test_detect_changes_empty_index(self, tmp_path):
        """Test detect_changes with an empty index."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        result = index.detect_changes([str(test_file)])

        assert str(test_file) in result["added"]
        assert len(result["deleted"]) == 0
        assert len(result["modified"]) == 0


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_index_with_unicode_paths(self, tmp_path):
        """Test handling of unicode characters in file paths."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        unicode_path = "/path/to/文件.py"
        metadata = FileMetadata(
            file_path=unicode_path,
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )

        index.update_file_metadata(unicode_path, metadata)
        index.save()

        # Reload and verify
        index2 = MetadataIndex(str(index_path))
        assert unicode_path in index2.metadata

    def test_index_with_special_characters_in_path(self, tmp_path):
        """Test handling of special characters in file paths."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        special_path = "/path/to/file with spaces & special!@#.py"
        metadata = FileMetadata(
            file_path=special_path,
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )

        index.update_file_metadata(special_path, metadata)
        index.save()

        # Reload and verify
        index2 = MetadataIndex(str(index_path))
        assert special_path in index2.metadata

    def test_very_large_index(self, tmp_path):
        """Test handling of a large number of tracked files."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Add 1000 files
        for i in range(1000):
            file_path = f"/path/to/file{i}.py"
            metadata = FileMetadata(
                file_path=file_path,
                mtime=1234567890.0 + i,
                size=1024 + i,
                chunk_count=i % 10,
            )
            index.update_file_metadata(file_path, metadata)

        index.save()

        # Reload and verify
        index2 = MetadataIndex(str(index_path))
        assert len(index2.metadata) == 1000
        assert "/path/to/file500.py" in index2.metadata

    def test_concurrent_save_operations(self, tmp_path):
        """Test that atomic writes prevent corruption."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Add initial data
        metadata = FileMetadata(
            file_path="/path/to/file.py",
            mtime=1234567890.0,
            size=1024,
            chunk_count=5,
        )
        index.update_file_metadata("/path/to/file.py", metadata)
        index.save()

        # Verify no temp file left behind
        temp_path = index_path.with_suffix(".tmp")
        assert not temp_path.exists()

        # Verify data integrity
        with open(index_path) as f:
            data = json.load(f)
        assert "/path/to/file.py" in data["files"]

    def test_io_error_during_load(self, tmp_path, capsys):
        """Test handling of IO errors during load by simulating read failure."""
        index_path = tmp_path / "metadata.json"

        # Create a file that exists but will fail when opened
        index_path.write_text('{"files": {}}')

        # Make the file unreadable (not the directory)
        index_path.chmod(0o000)

        try:
            # Should handle gracefully by starting fresh
            index = MetadataIndex(str(index_path))

            # Should start with empty state due to IOError
            assert index.metadata == {}

            captured = capsys.readouterr()
            assert "Corrupted metadata index" in captured.out
        finally:
            # Restore permissions
            index_path.chmod(0o644)

    def test_size_change_without_mtime_change(self, tmp_path):
        """Test detection when size changes but mtime is the same (edge case)."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        test_file = tmp_path / "test.py"
        test_file.write_text("content")
        stat = test_file.stat()

        # Track with wrong size but same mtime
        index.metadata[str(test_file)] = FileMetadata(
            file_path=str(test_file),
            mtime=stat.st_mtime,
            size=stat.st_size + 100,  # Wrong size
            chunk_count=1,
            content_hash="wrong_hash",
        )

        result = index.detect_changes([str(test_file)], verify_with_hash=False)

        # Should detect as modified due to size mismatch
        assert str(test_file) in result["modified"]


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestIntegration:
    """Integration-style tests that simulate realistic workflows."""

    def test_full_workflow_add_modify_delete(self, tmp_path):
        """Test a complete workflow of adding, modifying, and deleting files."""
        index_path = tmp_path / "metadata.json"

        # Create initial files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("print('hello')")
        file2.write_text("print('world')")

        # Initial indexing
        index = MetadataIndex(str(index_path))

        for f in [file1, file2]:
            stat = f.stat()
            index.update_file_metadata(
                str(f),
                FileMetadata(
                    file_path=str(f),
                    mtime=stat.st_mtime,
                    size=stat.st_size,
                    chunk_count=1,
                    content_hash=index.compute_file_hash(str(f)),
                ),
            )

        index.mark_reindex_complete()

        # Simulate time passing and changes
        time.sleep(0.01)

        # Modify file1
        file1.write_text("print('hello modified')")

        # Add file3
        file3 = tmp_path / "file3.py"
        file3.write_text("print('new file')")

        # Delete file2
        file2.unlink()

        # Detect changes
        current_files = [str(file1), str(file3)]
        result = index.detect_changes(current_files, verify_with_hash=True)

        assert str(file1) in result["modified"]
        assert str(file3) in result["added"]
        assert str(file2) in result["deleted"]

        # Verify persistence
        index.save()
        index2 = MetadataIndex(str(index_path))
        assert len(index2.metadata) == 2  # file1 and file2 (file2 not removed yet)

    def test_debounce_prevents_frequent_reindex(self, tmp_path):
        """Test that debounce prevents too frequent reindexing."""
        index_path = tmp_path / "metadata.json"
        index = MetadataIndex(str(index_path))

        # Mark reindex complete
        index.mark_reindex_complete()

        # Should not reindex immediately
        assert index.should_reindex(debounce_minutes=1) is False

        # Should reindex after enough time passes
        # (we fake this by modifying the timestamp)
        index.last_reindex_time = datetime.now().timestamp() - 120  # 2 minutes ago
        assert index.should_reindex(debounce_minutes=1) is True
