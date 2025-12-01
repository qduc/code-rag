"""Syntax-aware chunking using Tree-sitter."""

from typing import List, Dict, Optional, Tuple
import importlib
from tree_sitter import Language, Parser, Node

class SyntaxChunker:
    """
    Chunks code by respecting syntax boundaries (functions, classes, etc.).
    Uses tree-sitter to parse code and find logical split points.
    """

    # Mapping: language_key -> (module_name, function_name)
    LANGUAGE_PACKAGES = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "go": ("tree_sitter_go", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "java": ("tree_sitter_java", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
        "c": ("tree_sitter_c", "language"),
    }

    def __init__(self, chunk_size: int = 1024, overlap: int = 100):
        """
        Initialize the syntax chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks (not fully implemented for syntax chunking yet)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._parsers = {}

    def _get_parser(self, language_name: str):
        """Get or create a parser for the given language."""
        if language_name not in self._parsers:
            if language_name not in self.LANGUAGE_PACKAGES:
                return None

            module_name, func_name = self.LANGUAGE_PACKAGES[language_name]
            try:
                module = importlib.import_module(module_name)
                lang_func = getattr(module, func_name)
                # Create Language object from the capsule returned by the binding
                lang = Language(lang_func())
                parser = Parser()
                parser.language = lang
                self._parsers[language_name] = parser
            except ImportError:
                # Silently fail if language package is not installed
                return None
            except Exception as e:
                print(f"Error loading parser for {language_name}: {e}")
                return None
        return self._parsers[language_name]

    def chunk(self, content: str, language_name: str) -> List[Dict[str, any]]:
        """
        Chunk content using syntax-aware splitting.

        Args:
            content: The source code content
            language_name: The tree-sitter language identifier (e.g. 'python', 'javascript')

        Returns:
            List of dictionaries with 'text', 'start_byte', and 'end_byte' keys
        """
        parser = self._get_parser(language_name)
        if not parser:
            return []

        try:
            tree = parser.parse(bytes(content, "utf8"))
            return self._chunk_tree(tree.root_node, content)
        except Exception as e:
            print(f"Error parsing content for syntax chunking: {e}")
            return []

    def _chunk_tree(self, root_node: Node, content: str) -> List[Dict[str, any]]:
        """Process the syntax tree and generate chunks with byte offsets."""

        # 1. Flatten the tree into a stream of "atomic" nodes (small enough or irreducible)
        atoms = self._get_atoms(root_node)

        # 2. Fill in the gaps (whitespace/comments not covered by atoms) to get a continuous stream
        segments = self._create_segments(atoms, len(content))

        # 3. Group segments into chunks
        return self._group_segments(segments, content)

    def _get_atoms(self, node: Node) -> List[Node]:
        """
        Recursively collect nodes that are either:
        1. Smaller than chunk_size
        2. Have no children (leaf nodes)
        """
        node_len = node.end_byte - node.start_byte

        # If node fits in a chunk, or it's a leaf, treat it as an atom
        # Note: We check child_count because sometimes a large node might be a leaf (e.g. a huge string literal)
        if node_len < self.chunk_size or node.child_count == 0:
            return [node]

        # Otherwise, it's a large compound node. Recurse into children.
        atoms = []
        cursor = node.walk()
        if cursor.goto_first_child():
            while True:
                atoms.extend(self._get_atoms(cursor.node))
                if not cursor.goto_next_sibling():
                    break
        return atoms

    def _create_segments(self, atoms: List[Node], content_len: int) -> List[Tuple[int, int]]:
        """
        Create a continuous list of (start, end) ranges covering the whole content.
        Includes both the atoms and the gaps between them.
        """
        segments = []
        last_pos = 0

        for atom in atoms:
            # Add gap if exists
            if atom.start_byte > last_pos:
                segments.append((last_pos, atom.start_byte))

            # Add atom
            segments.append((atom.start_byte, atom.end_byte))
            last_pos = atom.end_byte

        # Add trailing gap
        if last_pos < content_len:
            segments.append((last_pos, content_len))

        return segments

    def _group_segments(self, segments: List[Tuple[int, int]], content: str) -> List[Dict[str, any]]:
        """Group segments into chunks respecting chunk_size, with byte offset tracking."""
        chunks = []
        current_chunk_segments = []
        current_chunk_len = 0

        for start, end in segments:
            segment_len = end - start

            # If adding this segment exceeds chunk size
            if current_chunk_len + segment_len > self.chunk_size:
                # If we have accumulated content, flush it
                if current_chunk_segments:
                    chunk_data = self._build_chunk_data(current_chunk_segments, content)
                    chunks.append(chunk_data)
                    current_chunk_segments = []
                    current_chunk_len = 0

                # Now handle the current segment
                if segment_len > self.chunk_size:
                    # This is a huge segment (e.g. a giant string literal or comment)
                    # We have to hard-split it
                    # For now, just add it as its own chunk (or multiple chunks)
                    # Simple fallback: just add it. The embedding model might truncate it.
                    # Or we could use the text-based chunker here recursively?
                    # Let's just add it for now to keep it simple.
                    chunks.append({
                        "text": content[start:end],
                        "start_byte": start,
                        "end_byte": end
                    })
                else:
                    # Start a new chunk with this segment
                    current_chunk_segments.append((start, end))
                    current_chunk_len = segment_len
            else:
                # Append to current chunk
                current_chunk_segments.append((start, end))
                current_chunk_len += segment_len

        # Flush remaining
        if current_chunk_segments:
            chunk_data = self._build_chunk_data(current_chunk_segments, content)
            chunks.append(chunk_data)

        return chunks

    def _build_chunk_data(self, segments: List[Tuple[int, int]], content: str) -> Dict[str, any]:
        """Reconstruct text from segments and return with byte offsets."""
        # Since segments are contiguous ranges from the original content,
        # we can technically just take content[first_start:last_end]
        # IF the segments are contiguous.
        # Our _create_segments ensures they are contiguous and cover the whole file.
        # But _group_segments might have split them.
        # Within a chunk, the segments should be contiguous.

        if not segments:
            return {"text": "", "start_byte": 0, "end_byte": 0}

        start = segments[0][0]
        end = segments[-1][1]
        return {
            "text": content[start:end],
            "start_byte": start,
            "end_byte": end
        }
