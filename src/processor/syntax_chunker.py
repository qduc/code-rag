"""Syntax-aware chunking using Tree-sitter."""

from typing import List, Dict, Optional, Tuple, Set
import importlib
from tree_sitter import Language, Parser, Node


# Node types that represent named definitions we want to extract
DEFINITION_NODE_TYPES = {
    # Python
    "function_definition",
    "class_definition",
    "decorated_definition",
    # JavaScript/TypeScript
    "function_declaration",
    "class_declaration",
    "method_definition",
    "arrow_function",
    "function",
    # Go
    "function_declaration",
    "method_declaration",
    "type_declaration",
    # Rust
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    # Java
    "method_declaration",
    "class_declaration",
    "interface_declaration",
    # C/C++
    "function_definition",
    "class_specifier",
    "struct_specifier",
}

# Node types that represent class/struct containers
CLASS_NODE_TYPES = {
    "class_definition",
    "class_declaration",
    "class_specifier",
    "struct_specifier",
    "struct_item",
    "impl_item",
    "interface_declaration",
    "trait_item",
}

# Node types that represent functions/methods
FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
    "arrow_function",
    "function",
    "function_item",
}


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
        self._current_tree = None  # Store current parse tree for metadata extraction

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
            List of dictionaries with 'text', 'start_byte', 'end_byte', and optional
            'function_name', 'class_name', 'symbol_type' keys
        """
        parser = self._get_parser(language_name)
        if not parser:
            return []

        try:
            tree = parser.parse(bytes(content, "utf8"))
            self._current_tree = tree  # Store for metadata extraction
            chunks = self._chunk_tree(tree.root_node, content)
            # Enrich chunks with AST metadata
            return self._enrich_chunks_with_metadata(chunks, tree.root_node, content)
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

    def _enrich_chunks_with_metadata(self, chunks: List[Dict[str, any]], root_node: Node, content: str) -> List[Dict[str, any]]:
        """
        Enrich chunks with AST metadata (function names, class names, symbol types).

        For each chunk, finds the enclosing or primary definition and extracts its name.
        """
        # Build an index of all definitions in the file
        definitions = self._collect_definitions(root_node, content)

        for chunk in chunks:
            start = chunk["start_byte"]
            end = chunk["end_byte"]

            # Find definitions that overlap with this chunk
            chunk_functions = []
            chunk_classes = []

            for defn in definitions:
                defn_start = defn["start_byte"]
                defn_end = defn["end_byte"]

                # Check if definition overlaps with chunk
                if defn_start < end and defn_end > start:
                    if defn["symbol_type"] == "function":
                        chunk_functions.append(defn["name"])
                    elif defn["symbol_type"] == "class":
                        chunk_classes.append(defn["name"])

            # Set metadata - use first/primary names found
            if chunk_functions:
                chunk["function_name"] = chunk_functions[0]
                if len(chunk_functions) > 1:
                    chunk["function_names"] = chunk_functions
            if chunk_classes:
                chunk["class_name"] = chunk_classes[0]
                if len(chunk_classes) > 1:
                    chunk["class_names"] = chunk_classes

            # Determine primary symbol type
            if chunk_functions and chunk_classes:
                chunk["symbol_type"] = "method"  # Function inside a class
            elif chunk_functions:
                chunk["symbol_type"] = "function"
            elif chunk_classes:
                chunk["symbol_type"] = "class"

        return chunks

    def _collect_definitions(self, node: Node, content: str, parent_class: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Recursively collect all function and class definitions from the AST.

        Returns a list of dicts with: name, symbol_type, start_byte, end_byte, parent_class
        """
        definitions = []
        node_type = node.type

        # Check if this node is a definition
        if node_type in DEFINITION_NODE_TYPES:
            name = self._extract_definition_name(node, content)
            if name:
                if node_type in CLASS_NODE_TYPES:
                    symbol_type = "class"
                elif node_type in FUNCTION_NODE_TYPES:
                    symbol_type = "function"
                else:
                    symbol_type = "definition"

                definitions.append({
                    "name": name,
                    "symbol_type": symbol_type,
                    "start_byte": node.start_byte,
                    "end_byte": node.end_byte,
                    "parent_class": parent_class,
                })

                # Update parent_class for nested definitions
                if symbol_type == "class":
                    parent_class = name

        # Recurse into children
        for child in node.children:
            definitions.extend(self._collect_definitions(child, content, parent_class))

        return definitions

    def _extract_definition_name(self, node: Node, content: str) -> Optional[str]:
        """
        Extract the name of a function/class definition from an AST node.

        Handles various language patterns for extracting identifier names.
        """
        # Handle decorated definitions (Python)
        if node.type == "decorated_definition":
            # Find the actual definition inside
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    return self._extract_definition_name(child, content)
            return None

        # Look for 'name' or 'identifier' child nodes
        for child in node.children:
            if child.type in ("identifier", "name", "property_identifier"):
                return content[child.start_byte:child.end_byte]
            # Handle typed parameters pattern (TypeScript)
            if child.type == "type_identifier":
                return content[child.start_byte:child.end_byte]

        # Fallback: look for first identifier in any child
        for child in node.children:
            if child.type == "identifier":
                return content[child.start_byte:child.end_byte]
            # Recurse one level for compound patterns
            for grandchild in child.children:
                if grandchild.type in ("identifier", "name"):
                    return content[grandchild.start_byte:grandchild.end_byte]

        return None
