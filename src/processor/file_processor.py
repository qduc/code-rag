"""File processor for discovering and reading source code files."""

import fnmatch
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .syntax_chunker import SyntaxChunker

# Patterns for test files that can be optionally excluded
TEST_FILE_PATTERNS = [
    "test_*.py",
    "*_test.py",
    "*_test.go",
    "*_test.js",
    "*_test.ts",
    "*.spec.js",
    "*.spec.ts",
    "*.test.js",
    "*.test.ts",
    "*.test.jsx",
    "*.test.tsx",
    "Test*.java",
    "*Test.java",
    "*Tests.java",
    "*_test.rb",
    "test_*.rb",
    "*_spec.rb",
]

# Patterns for test directories
TEST_DIR_PATTERNS = [
    "test",
    "tests",
    "testing",
    "__tests__",
    "spec",
    "specs",
]


def byte_offset_to_line_number(content: str, byte_offset: int) -> int:
    """
    Convert a byte offset to a line number.

    Args:
        content: The full text content
        byte_offset: Byte position in the content

    Returns:
        Line number (1-indexed)
    """
    # Count newlines up to the byte offset
    return content[:byte_offset].count("\n") + 1


class FileProcessor:
    """Handles file discovery and reading for codebase processing."""

    # Common source code file extensions
    SOURCE_EXTENSIONS = {
        # Programming languages
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".clj",
        ".lua",
        ".r",
        ".m",
        ".mm",
        ".pl",
        ".pm",
        ".sh",
        ".bash",
        ".zsh",
        # Web
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        ".vue",
        ".svelte",
        # Data/Config
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".ini",
        ".cfg",
        ".conf",
        # Documentation
        ".md",
        ".rst",
        ".txt",
        # Build/DevOps
        ".dockerfile",
        ".makefile",
        ".gradle",
        ".cmake",
    }

    # Mapping from extension to tree-sitter language name
    EXTENSION_TO_LANGUAGE = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "c_sharp",
    }

    def __init__(
        self,
        ignore_patterns: List[str] = None,
        additional_ignore_patterns: List[str] = None,
        exclude_tests: bool = False,
        include_file_header: bool = True,
    ):
        """
        Initialize the file processor.

        Args:
            ignore_patterns: List of patterns to ignore during file discovery
            additional_ignore_patterns: Additional patterns to ignore (besides defaults)
            exclude_tests: Whether to exclude test files from processing
            include_file_header: Whether to include file headers in chunks
        """
        self.ignore_patterns = ignore_patterns or self._default_ignore_patterns()
        if additional_ignore_patterns:
            self.ignore_patterns.extend(additional_ignore_patterns)
        self.exclude_tests = exclude_tests
        self.include_file_header = include_file_header
        self._gitignore_patterns: Set[str] = set()
        self.syntax_chunker = SyntaxChunker(include_file_header=include_file_header)

    def _default_ignore_patterns(self) -> List[str]:
        """Return default patterns to ignore."""
        return [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".egg-info",
            "dist",
            "build",
            ".code-rag",
            ".idea",
            ".vscode",
            "*.pyc",
            "*.pyo",
            "*.egg",
            "*.whl",
            ".DS_Store",
            "Thumbs.db",
            # Dependency lock files
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            "poetry.lock",
            "Pipfile.lock",
            "Cargo.lock",
            "go.sum",
            "Gemfile.lock",
            # Build artifacts and minified code
            "*.min.js",
            "*.min.css",
            "*.map",
            # Log files
            "*.log",
        ]

    def _load_gitignore(self, root_path: str) -> None:
        """
        Load .gitignore patterns from the root directory.

        Args:
            root_path: Root directory containing .gitignore
        """
        gitignore_path = Path(root_path) / ".gitignore"
        if gitignore_path.exists():
            try:
                with open(gitignore_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith("#"):
                            self._gitignore_patterns.add(line)
            except Exception:
                pass  # Silently ignore gitignore parsing errors

    def _should_ignore(self, path: Path, root_path: Path) -> bool:
        """
        Check if a path should be ignored based on patterns.

        Args:
            path: The path to check
            root_path: The root path for relative path calculation

        Returns:
            True if the path should be ignored
        """
        # Get relative path for pattern matching
        try:
            rel_path = path.relative_to(root_path)
        except ValueError:
            rel_path = path

        rel_path_str = str(rel_path)
        path_parts = rel_path.parts

        # Check default ignore patterns
        for pattern in self.ignore_patterns:
            # Check if any path component matches the pattern
            for part in path_parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            # Check full relative path
            if fnmatch.fnmatch(rel_path_str, pattern):
                return True

        # Check gitignore patterns
        for pattern in self._gitignore_patterns:
            # Handle directory patterns (ending with /)
            if pattern.endswith("/"):
                dir_pattern = pattern[:-1]
                for part in path_parts:
                    if fnmatch.fnmatch(part, dir_pattern):
                        return True
            else:
                # Check against path components and full path
                for part in path_parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True
                if fnmatch.fnmatch(rel_path_str, pattern):
                    return True
                # Handle patterns with wildcards
                if fnmatch.fnmatch(path.name, pattern):
                    return True

        return False

    def _is_test_file(self, path: Path) -> bool:
        """
        Check if a file is a test file based on common naming patterns.

        Args:
            path: Path to the file

        Returns:
            True if it's a test file
        """
        filename = path.name

        # Check test file patterns
        for pattern in TEST_FILE_PATTERNS:
            if fnmatch.fnmatch(filename, pattern):
                return True

        # Check if in a test directory
        for part in path.parts:
            for dir_pattern in TEST_DIR_PATTERNS:
                if part.lower() == dir_pattern:
                    return True

        return False

    def _is_source_file(self, path: Path) -> bool:
        """
        Check if a file is a source code file.

        Args:
            path: Path to the file

        Returns:
            True if it's a source code file
        """
        # Check extension
        suffix = path.suffix.lower()
        if suffix in self.SOURCE_EXTENSIONS:
            return True

        # Check special filenames without extensions
        name_lower = path.name.lower()
        if name_lower in {
            "makefile",
            "dockerfile",
            "vagrantfile",
            "gemfile",
            "rakefile",
        }:
            return True

        return False

    def discover_files(self, root_path: str) -> List[str]:
        """
        Recursively discover all relevant source code files.

        Args:
            root_path: Root directory to start discovery from

        Returns:
            List of file paths discovered
        """
        root = Path(root_path).resolve()
        self._load_gitignore(root_path)

        discovered_files = []

        for dirpath, dirnames, filenames in os.walk(root):
            current_dir = Path(dirpath)

            # Filter out ignored directories (modifying dirnames in-place)
            dirnames[:] = [
                d for d in dirnames if not self._should_ignore(current_dir / d, root)
            ]

            # Optionally filter out test directories
            if self.exclude_tests:
                dirnames[:] = [
                    d for d in dirnames if d.lower() not in TEST_DIR_PATTERNS
                ]

            # Process files
            for filename in filenames:
                file_path = current_dir / filename

                # Skip ignored files
                if self._should_ignore(file_path, root):
                    continue

                # Optionally skip test files
                if self.exclude_tests and self._is_test_file(file_path):
                    continue

                # Only include source files
                if self._is_source_file(file_path):
                    discovered_files.append(str(file_path))

        return sorted(discovered_files)

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read file contents, handling different encodings.

        Args:
            file_path: Path to the file to read

        Returns:
            File contents as string, or None if reading fails
        """
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                # Log error and return None for other exceptions
                print(f"Error reading {file_path}: {e}")
                return None

        print(f"Could not decode {file_path} with any supported encoding")
        return None

    def chunk_file(
        self,
        content: str,
        chunk_size: int = 1024,
    ) -> List[str]:
        """
        Split file content into chunks.

        Uses a naive character-based approach, trying to split at line boundaries.

        Args:
            content: File content to chunk
            chunk_size: Target size of each chunk in characters

        Returns:
            List of chunks
        """
        if chunk_size is None or chunk_size <= 0:
            chunk_size = 1

        if not content:
            return []

        if len(content) <= chunk_size:
            return [content]

        chunks = []
        start = 0

        while start < len(content):
            # Calculate end position
            end = start + chunk_size

            if end >= len(content):
                # Last chunk
                chunks.append(content[start:])
                break

            # Try to find a good break point (newline) near the end
            # Search backwards from end for a newline
            break_point = content.rfind("\n", start + chunk_size // 2, end)

            if break_point == -1:
                # No newline found, just use the chunk_size
                break_point = end
            else:
                # Include the newline in the current chunk
                break_point += 1

            chunks.append(content[start:break_point])

            # Move start position to the next block
            start = break_point

        return chunks

    def _sanitize_chunk_parameters(self, chunk_size: Optional[int]) -> int:
        """Ensure chunk size is a sane positive number."""
        base_size = (
            chunk_size
            if chunk_size and chunk_size > 0
            else self.syntax_chunker.chunk_size or 1
        )
        return max(1, int(base_size))

    def process_file(
        self,
        file_path: str,
        chunk_size: int = 1024,
    ) -> List[Dict[str, Any]]:
        """
        Process a single file: read, chunk, and prepare for embedding.

        Args:
            file_path: Path to the file to process
            chunk_size: Size of chunks to create

        Returns:
            List of dictionaries with chunk data and metadata
        """
        content = self.read_file(file_path)
        if content is None:
            return []

        # Determine language and try syntax chunking
        ext = Path(file_path).suffix.lower()
        language = self.EXTENSION_TO_LANGUAGE.get(ext)

        safe_chunk_size = self._sanitize_chunk_parameters(chunk_size)

        chunks = []
        if language:
            # Update chunk size for this request
            self.syntax_chunker.chunk_size = safe_chunk_size
            chunks = self.syntax_chunker.chunk(content, language)

        # Fallback to basic chunking if syntax chunking failed or not supported
        if not chunks:
            chunks = self.chunk_file(content, safe_chunk_size)

        result = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            # Handle both dict (from syntax chunker) and str (from basic chunker)
            if isinstance(chunk, dict):
                chunk_text = chunk["text"]
                start_byte = chunk["start_byte"]
                end_byte = chunk["end_byte"]
                start_line = byte_offset_to_line_number(content, start_byte)
                end_line = byte_offset_to_line_number(content, end_byte)
            else:
                # Basic chunking - calculate position manually
                chunk_text = chunk
                # Find position of this chunk in content
                start_byte = content.find(chunk_text)
                if start_byte == -1:
                    start_byte = 0
                end_byte = start_byte + len(chunk_text)
                start_line = byte_offset_to_line_number(content, start_byte)
                end_line = byte_offset_to_line_number(content, end_byte)

            # Build metadata dict
            metadata = {
                "file_path": file_path,
                "chunk_index": i,
                "total_chunks": total_chunks,
                "start_line": start_line,
                "end_line": end_line,
                "start_byte": start_byte,
                "end_byte": end_byte,
                # Adjacency metadata for chunk traversal
                "prev_id": i - 1 if i > 0 else -1,  # -1 indicates no previous
                "next_id": (
                    i + 1 if i < total_chunks - 1 else -1
                ),  # -1 indicates no next
            }

            # Add AST metadata if available (from syntax chunker)
            if isinstance(chunk, dict):
                if "function_name" in chunk:
                    metadata["function_name"] = chunk["function_name"]
                if "class_name" in chunk:
                    metadata["class_name"] = chunk["class_name"]
                if "symbol_type" in chunk:
                    metadata["symbol_type"] = chunk["symbol_type"]
                if chunk.get("has_file_header"):
                    metadata["has_file_header"] = True
                if chunk.get("is_continuation"):
                    metadata["is_continuation"] = True
                if chunk.get("has_signature_context"):
                    metadata["has_signature_context"] = True

            result.append(
                {
                    "id": f"{file_path}:chunk_{i}",
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

        return result

    def get_file_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file statistics (mtime, size).

        Args:
            file_path: Path to the file

        Returns:
            Dict with 'mtime' and 'size', or None on error
        """
        try:
            stat = os.stat(file_path)
            return {"mtime": stat.st_mtime, "size": stat.st_size}
        except OSError as e:
            print(f"Error getting stats for {file_path}: {e}")
            return None

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
