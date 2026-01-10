"""Syntax-aware chunking using Tree-sitter."""

import importlib
import logging
from typing import Dict, List, Optional, Tuple

from tree_sitter import Language, Node, Parser

# Get logger for this module
logger = logging.getLogger(__name__)


# Node types that represent file-level header items (imports, constants, type definitions)
HEADER_NODE_TYPES = {
    # Python
    "import_statement",
    "import_from_statement",
    "future_import_statement",
    # JavaScript/TypeScript
    # "import_statement",  # Duplicate of Python
    "import_declaration",
    "export_statement",
    # Go
    # "import_declaration",  # Duplicate of JS/TS
    "package_clause",
    # Rust
    "use_declaration",
    "extern_crate_declaration",
    # Java
    # "import_declaration",  # Duplicate of JS/TS
    "package_declaration",
    # C/C++
    "preproc_include",
    "preproc_define",
    "type_definition",
}

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
    # "function_declaration",  # Duplicate of JS/TS
    "method_declaration",
    "type_declaration",
    # Rust
    "function_item",
    "impl_item",
    "struct_item",
    "enum_item",
    "trait_item",
    # Java
    # "method_declaration",  # Duplicate of Go
    # "class_declaration",  # Duplicate of JS/TS
    "interface_declaration",
    # C/C++
    # "function_definition",  # Duplicate of Python
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

    def __init__(self, chunk_size: int = 1024, include_file_header: bool = True):
        """
        Initialize the syntax chunker.

        Args:
            chunk_size: Target size of each chunk in characters
            include_file_header: Whether to extract and include file headers (imports, etc.)
        """
        self.chunk_size = chunk_size
        self.include_file_header = include_file_header
        self._parsers = {}
        self._file_header: Optional[str] = None  # Cache extracted file header
        self._file_header_end_byte: int = 0  # End byte of file header

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
            'function_name', 'class_name', 'symbol_type', 'file_header', 'prev_id', 'next_id' keys
        """
        self._ensure_valid_parameters()

        parser = self._get_parser(language_name)
        if not parser:
            return []

        try:
            tree = parser.parse(bytes(content, "utf8"))

            # Extract file header (imports, constants, etc.) if enabled
            if self.include_file_header:
                self._file_header, self._file_header_end_byte = (
                    self._extract_file_header(tree.root_node, content)
                )
            else:
                self._file_header = None
                self._file_header_end_byte = 0

            chunks = self._chunk_tree(tree.root_node, content)
            # Enrich chunks with AST metadata and adjacency info
            chunks = self._enrich_chunks_with_metadata(chunks, tree.root_node, content)
            chunks = self._add_adjacency_metadata(chunks)
            return chunks
        except Exception as e:
            print(f"Error parsing content for syntax chunking: {e}")
            return []

    def _ensure_valid_parameters(self) -> None:
        """Clamp chunker settings to safe values to avoid infinite loops."""
        if self.chunk_size < 1:
            self.chunk_size = 1

    def _extract_file_header(
        self, root_node: Node, content: str
    ) -> Tuple[Optional[str], int]:
        """
        Extract file-level header content (imports, constants, type definitions).

        Collects contiguous header statements from the beginning of the file.

        Args:
            root_node: Root node of the parse tree
            content: Full source code content

        Returns:
            Tuple of (header_text, end_byte) or (None, 0) if no header found
        """
        header_nodes = []
        last_header_end = 0

        for child in root_node.children:
            # Check if this is a header-type node
            if child.type in HEADER_NODE_TYPES:
                header_nodes.append(child)
                last_header_end = child.end_byte
            elif child.type in ("comment", "line_comment", "block_comment"):
                # Include leading comments as part of header
                if not header_nodes or child.start_byte <= last_header_end + 10:
                    header_nodes.append(child)
                    last_header_end = child.end_byte
            elif header_nodes:
                # Stop at first non-header, non-comment node after we've seen headers
                break

        if not header_nodes:
            return None, 0

        # Extract header text
        start = header_nodes[0].start_byte
        end = header_nodes[-1].end_byte
        header_text = content[start:end].strip()

        return header_text, end

    def _add_adjacency_metadata(
        self, chunks: List[Dict[str, any]]
    ) -> List[Dict[str, any]]:
        """
        Add prev_id and next_id fields to each chunk for adjacency traversal.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Chunks with added prev_id and next_id fields
        """
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk["chunk_index"] = i
            chunk["total_chunks"] = total
            chunk["prev_id"] = i - 1 if i > 0 else None
            chunk["next_id"] = i + 1 if i < total - 1 else None

            # Include file header reference if available
            if self._file_header:
                chunk["has_file_header"] = True

        return chunks

    def get_file_header(self) -> Optional[str]:
        """
        Get the extracted file header from the last chunked file.

        Returns:
            The file header text (imports, constants, etc.) or None if not available
        """
        return self._file_header

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

    def _create_segments(
        self, atoms: List[Node], content_len: int
    ) -> List[Tuple[int, int]]:
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

    def _group_segments(
        self, segments: List[Tuple[int, int]], content: str
    ) -> List[Dict[str, any]]:
        """Group segments into chunks respecting chunk_size while tracking byte offsets."""
        logger.debug(
            f"_group_segments: Processing {len(segments)} segments, content length: {len(content)}"
        )
        chunks = []
        current_chunk_segments = []
        current_chunk_len = 0
        iteration_count = 0

        for start, end in segments:
            iteration_count += 1
            if iteration_count % 100 == 0:
                logger.debug(
                    f"_group_segments: iteration {iteration_count}, chunks created: {len(chunks)}"
                )
            logger.debug(f"  Processing segment [{start}:{end}], length: {end-start}")
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
                    # This is a huge segment - try to extract signature for preservation
                    sub_chunks = self._split_large_segment(start, end, content)
                    chunks.extend(sub_chunks)
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

    def _split_large_segment(
        self, start: int, end: int, content: str
    ) -> List[Dict[str, any]]:
        """
        Split a large segment (e.g., huge function) into smaller chunks.

        Preserves function signature and docstring in each sub-chunk for context.

        Args:
            start: Start byte of the segment
            end: End byte of the segment
            content: Full source content
        Returns:
            List of chunk dictionaries
        """
        logger.debug(
            f"_split_large_segment: Splitting segment [{start}:{end}], length: {end-start}"
        )
        logger.debug(f"  chunk_size={self.chunk_size}")
        segment_text = content[start:end]
        chunks = []
        iteration_count = 0

        # Try to extract signature + docstring from the segment
        signature_end = self._find_signature_end(segment_text)
        signature = segment_text[:signature_end] if signature_end > 0 else ""

        # Calculate effective chunk size (accounting for signature preservation)
        effective_chunk_size = self.chunk_size - len(signature)
        if effective_chunk_size < 200:
            # If too little space, just use basic splitting without signature
            effective_chunk_size = self.chunk_size
            signature = ""

        # Split the body into chunks
        body_start = signature_end
        current_pos = body_start

        logger.debug(
            f"  Starting split loop: body_start={body_start}, segment_text length={len(segment_text)}, "
            f"effective_chunk_size={effective_chunk_size}"
        )

        while current_pos < len(segment_text):
            iteration_count += 1
            if iteration_count > 1000:
                logger.error(
                    "INFINITE LOOP DETECTED in _split_large_segment after 1000 iterations!"
                )
                logger.error(
                    f"  current_pos={current_pos}, len(segment_text)={len(segment_text)}"
                )
                logger.error(f"  effective_chunk_size={effective_chunk_size}")
                logger.error(f"  chunks created so far: {len(chunks)}")
                break

            if iteration_count % 10 == 0:
                logger.debug(
                    f"  Split iteration {iteration_count}: current_pos={current_pos}/{len(segment_text)}, "
                    f"chunks={len(chunks)}"
                )

            chunk_end = min(current_pos + effective_chunk_size, len(segment_text))

            # Try to find a good break point (newline)
            if chunk_end < len(segment_text):
                break_point = segment_text.rfind(
                    "\n", current_pos + effective_chunk_size // 2, chunk_end
                )
                if break_point > current_pos:
                    chunk_end = break_point + 1

            # Build chunk text with preserved context
            chunk_parts = []
            if signature and chunks:  # Subsequent chunks get signature prepended
                chunk_parts.append(
                    f"# [continued from above]\n{signature}\n    # ...\n"
                )

            body_text = segment_text[current_pos:chunk_end]
            chunk_parts.append(body_text)
            chunk_text = "".join(chunk_parts)

            # Calculate actual byte positions
            chunk_start_byte = start + current_pos
            chunk_end_byte = start + chunk_end

            chunks.append(
                {
                    "text": chunk_text,
                    "start_byte": chunk_start_byte,
                    "end_byte": chunk_end_byte,
                    "is_continuation": len(chunks) > 0,
                    "has_signature_context": bool(signature) and len(chunks) > 0,
                }
            )

            # Move to next chunk
            old_pos = current_pos
            current_pos = chunk_end
            logger.debug(
                f"  Moving position: {old_pos} -> {current_pos} (chunk_end={chunk_end})"
            )

        logger.debug(f"_split_large_segment: Completed, created {len(chunks)} chunks")
        return chunks

    def _find_signature_end(self, text: str) -> int:
        """
        Find the end of a function/class signature including docstring.

        Looks for patterns like:
        - def func_name(...):
        -     '''docstring'''
        - class ClassName:
        -     '''docstring'''

        Returns:
            Byte position where signature+docstring ends, or 0 if not found
        """
        lines = text.split("\n")
        if not lines:
            return 0

        # Find the first line with a definition
        signature_lines = []
        in_docstring = False
        docstring_delim = None

        for line in lines:
            stripped = line.strip()

            if not signature_lines:
                # Looking for start of definition
                if stripped.startswith(
                    ("def ", "class ", "async def ", "fn ", "func ", "function ")
                ):
                    signature_lines.append(line)
                    # Check if definition ends on this line
                    if (
                        stripped.endswith(":")
                        or stripped.endswith("{")
                        or stripped.endswith(")")
                    ):
                        continue
                elif stripped.startswith(
                    ("public ", "private ", "protected ", "static ")
                ):
                    signature_lines.append(line)
                else:
                    continue
            elif not in_docstring:
                # Already have signature, look for docstring
                if stripped.startswith(('"""', "'''", "/*", "//")):
                    in_docstring = True
                    if stripped.startswith('"""'):
                        docstring_delim = '"""'
                    elif stripped.startswith("'''"):
                        docstring_delim = "'''"
                    signature_lines.append(line)
                    # Check if docstring ends on same line
                    if stripped.count(docstring_delim) >= 2:
                        in_docstring = False
                        break
                elif stripped and not stripped.startswith("#"):
                    # Hit actual code, stop
                    break
                else:
                    signature_lines.append(line)
            else:
                # Inside docstring
                signature_lines.append(line)
                if docstring_delim and docstring_delim in stripped:
                    in_docstring = False
                    break

            # Limit signature extraction to reasonable size
            if len(signature_lines) > 20:
                break

        if not signature_lines:
            return 0

        # Calculate byte position
        result = "\n".join(signature_lines)
        return len(result) + 1  # +1 for the newline

    def _build_chunk_data(
        self, segments: List[Tuple[int, int]], content: str
    ) -> Dict[str, any]:
        """
        Reconstruct text from segments and return with byte offsets.

        Args:
            segments: List of (start, end) byte positions
            content: Full source content

        Returns:
            Dictionary with text, start_byte, end_byte
        """
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
        chunk_text = content[start:end]

        return {
            "text": chunk_text,
            "start_byte": start,
            "end_byte": end,
        }

    def _enrich_chunks_with_metadata(
        self, chunks: List[Dict[str, any]], root_node: Node, content: str
    ) -> List[Dict[str, any]]:
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

    def _collect_definitions(
        self, node: Node, content: str, parent_class: Optional[str] = None
    ) -> List[Dict[str, any]]:
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

                definitions.append(
                    {
                        "name": name,
                        "symbol_type": symbol_type,
                        "start_byte": node.start_byte,
                        "end_byte": node.end_byte,
                        "parent_class": parent_class,
                    }
                )

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
                return content[child.start_byte : child.end_byte]
            # Handle typed parameters pattern (TypeScript)
            if child.type == "type_identifier":
                return content[child.start_byte : child.end_byte]

        # Fallback: look for first identifier in any child
        for child in node.children:
            if child.type == "identifier":
                return content[child.start_byte : child.end_byte]
            # Recurse one level for compound patterns
            for grandchild in child.children:
                if grandchild.type in ("identifier", "name"):
                    return content[grandchild.start_byte : grandchild.end_byte]

        return None
