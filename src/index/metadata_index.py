"""Metadata index manager for tracking file changes."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class FileMetadata:
    """Metadata for a single file in the codebase."""

    file_path: str
    mtime: float
    size: int
    chunk_count: int
    content_hash: Optional[str] = None
    last_indexed: float = 0.0  # Unix timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "FileMetadata":
        """Create from dictionary."""
        return FileMetadata(**data)


class MetadataIndex:
    """Manages metadata index for tracking file changes."""

    def __init__(self, index_path: str):
        """
        Initialize metadata index.

        Args:
            index_path: Path to the metadata index file (JSON)
        """
        self.index_path = Path(index_path)
        self.metadata: Dict[str, FileMetadata] = {}
        self.last_reindex_time: float = 0.0
        self._load()

    def _load(self) -> None:
        """Load metadata from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                    self.last_reindex_time = data.get("last_reindex_time", 0.0)
                    files = data.get("files", {})
                    self.metadata = {
                        path: FileMetadata.from_dict(meta)
                        for path, meta in files.items()
                    }
            except (json.JSONDecodeError, IOError) as e:
                # Corrupted index - start fresh
                print(f"Warning: Corrupted metadata index, starting fresh: {e}")
                self.metadata = {}
                self.last_reindex_time = 0.0

    def _save(self) -> None:
        """Save metadata to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_reindex_time": self.last_reindex_time,
            "files": {path: meta.to_dict() for path, meta in self.metadata.items()},
        }
        # Atomic write using temporary file
        temp_path = self.index_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.index_path)

    def get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get metadata for a file."""
        return self.metadata.get(file_path)

    def update_file_metadata(self, file_path: str, metadata: FileMetadata) -> None:
        """Update metadata for a file."""
        self.metadata[file_path] = metadata

    def remove_file_metadata(self, file_path: str) -> None:
        """Remove metadata for a file."""
        self.metadata.pop(file_path, None)

    def get_all_tracked_files(self) -> Set[str]:
        """Get set of all tracked file paths."""
        return set(self.metadata.keys())

    def mark_reindex_complete(self) -> None:
        """Update last reindex timestamp to now."""
        self.last_reindex_time = datetime.now().timestamp()
        self._save()

    def should_reindex(self, debounce_minutes: int) -> bool:
        """
        Check if enough time has passed since last reindex.

        Args:
            debounce_minutes: Minimum minutes between reindexing

        Returns:
            True if reindex should be performed
        """
        if self.last_reindex_time == 0.0:
            return True  # Never indexed before

        elapsed = datetime.now().timestamp() - self.last_reindex_time
        elapsed_minutes = elapsed / 60
        return elapsed_minutes >= debounce_minutes

    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """
        Compute SHA256 hash of file content.

        Args:
            file_path: Path to the file

        Returns:
            Hex digest of file hash, or None on error
        """
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            print(f"Error computing hash for {file_path}: {e}")
            return None

    def detect_changes(
        self, current_files: List[str], verify_with_hash: bool = True
    ) -> Dict[str, List[str]]:
        """
        Detect which files have changed, been added, or deleted.

        Args:
            current_files: List of currently discovered files
            verify_with_hash: If True, verify mtime/size changes with content hash

        Returns:
            Dict with keys:
                - 'added': New files not in index
                - 'modified': Files with changed mtime/size/hash
                - 'deleted': Files in index but not on disk
                - 'unchanged': Files with no changes
        """
        result = {"added": [], "modified": [], "deleted": [], "unchanged": []}

        current_files_set = set(current_files)
        tracked_files = self.get_all_tracked_files()

        # Detect deleted files
        deleted = tracked_files - current_files_set
        result["deleted"] = list(deleted)

        # Check each current file
        for file_path in current_files:
            try:
                stat = os.stat(file_path)
                current_mtime = stat.st_mtime
                current_size = stat.st_size

                metadata = self.get_file_metadata(file_path)

                if metadata is None:
                    # New file
                    result["added"].append(file_path)
                else:
                    # Check if mtime or size changed
                    if metadata.mtime != current_mtime or metadata.size != current_size:

                        # Verify actual content change with hash if requested
                        if verify_with_hash:
                            current_hash = self.compute_file_hash(file_path)
                            if current_hash and current_hash != metadata.content_hash:
                                result["modified"].append(file_path)
                            else:
                                # Hash matches - just mtime/size false positive
                                result["unchanged"].append(file_path)
                        else:
                            # Trust mtime/size without verification
                            result["modified"].append(file_path)
                    else:
                        # No changes
                        result["unchanged"].append(file_path)

            except OSError as e:
                # File disappeared or permission error
                print(f"Error checking {file_path}: {e}")
                result["deleted"].append(file_path)

        return result

    def save(self) -> None:
        """Explicitly save metadata to disk."""
        self._save()
