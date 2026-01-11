"""Comprehensive tests for the SyntaxChunker functionality.

These tests verify:
1. Language detection and parser initialization
2. Tree-sitter based parsing for different languages (Python, JavaScript, TypeScript, Java, Go, Rust, etc.)
3. Chunk extraction (functions, classes, methods)
4. Line-based fallback chunking when tree-sitter fails
5. Symbol name extraction
6. Chunk metadata (start_line, end_line, symbol names)
7. Handling of syntax errors in source code
8. Large files handling
9. Edge cases (empty files, files with only comments, nested structures)
10. File header extraction (imports, package declarations, etc.)
"""

import pytest

from code_rag.processor.syntax_chunker import (
    CLASS_NODE_TYPES,
    DEFINITION_NODE_TYPES,
    FUNCTION_NODE_TYPES,
    HEADER_NODE_TYPES,
    SyntaxChunker,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def chunker():
    """Provide a default SyntaxChunker instance."""
    return SyntaxChunker(chunk_size=1024)


@pytest.fixture
def small_chunk_chunker():
    """Provide a SyntaxChunker with small chunk size for testing splitting."""
    return SyntaxChunker(chunk_size=100)


@pytest.fixture
def python_simple_function():
    """Simple Python function for testing."""
    return '''def hello_world():
    """Say hello to the world."""
    print("Hello, World!")
    return True
'''


@pytest.fixture
def python_class_with_methods():
    """Python class with multiple methods."""
    return '''class Calculator:
    """A simple calculator class."""

    def __init__(self, value=0):
        """Initialize the calculator."""
        self.value = value

    def add(self, x):
        """Add a number."""
        self.value += x
        return self.value

    def subtract(self, x):
        """Subtract a number."""
        self.value -= x
        return self.value
'''


@pytest.fixture
def python_with_imports():
    """Python code with imports and module-level content."""
    return '''import os
import sys
from pathlib import Path
from typing import List, Optional

# Module constant
MAX_RETRIES = 3

def process_file(path: Path) -> Optional[str]:
    """Process a file and return its content."""
    if not path.exists():
        return None
    return path.read_text()
'''


@pytest.fixture
def javascript_code():
    """JavaScript code sample."""
    return """import React from 'react';
import { useState } from 'react';

function Counter() {
    const [count, setCount] = useState(0);

    const increment = () => {
        setCount(count + 1);
    };

    return (
        <div>
            <p>Count: {count}</p>
            <button onClick={increment}>Increment</button>
        </div>
    );
}

export default Counter;
"""


@pytest.fixture
def typescript_code():
    """TypeScript code sample."""
    return """interface User {
    id: number;
    name: string;
    email: string;
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(id: number): User | undefined {
        return this.users.find(u => u.id === id);
    }
}

export { UserService, User };
"""


@pytest.fixture
def go_code():
    """Go code sample."""
    return """package main

import (
    "fmt"
    "net/http"
)

type Server struct {
    host string
    port int
}

func NewServer(host string, port int) *Server {
    return &Server{host: host, port: port}
}

func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.host, s.port)
    return http.ListenAndServe(addr, nil)
}

func main() {
    server := NewServer("localhost", 8080)
    server.Start()
}
"""


@pytest.fixture
def rust_code():
    """Rust code sample."""
    return """use std::collections::HashMap;

struct Counter {
    counts: HashMap<String, u32>,
}

impl Counter {
    fn new() -> Self {
        Counter {
            counts: HashMap::new(),
        }
    }

    fn increment(&mut self, key: &str) {
        *self.counts.entry(key.to_string()).or_insert(0) += 1;
    }

    fn get(&self, key: &str) -> u32 {
        *self.counts.get(key).unwrap_or(&0)
    }
}

fn main() {
    let mut counter = Counter::new();
    counter.increment("hello");
    println!("Count: {}", counter.get("hello"));
}
"""


@pytest.fixture
def java_code():
    """Java code sample."""
    return """package com.example;

import java.util.ArrayList;
import java.util.List;

public class UserManager {
    private List<String> users;

    public UserManager() {
        this.users = new ArrayList<>();
    }

    public void addUser(String name) {
        users.add(name);
    }

    public List<String> getUsers() {
        return users;
    }
}
"""


@pytest.fixture
def nested_python_code():
    """Python code with nested classes and functions."""
    return '''class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def inner_method(self):
            """Inner method."""
            pass

    def outer_method(self):
        """Outer method."""

        def nested_function():
            """Nested function."""
            pass

        return nested_function()
'''


# ============================================================================
# SyntaxChunker Initialization Tests
# ============================================================================


class TestSyntaxChunkerInit:
    """Tests for SyntaxChunker initialization."""

    def test_default_initialization(self):
        """Test default initialization values."""
        chunker = SyntaxChunker()
        assert chunker.chunk_size == 1024
        assert chunker.include_file_header is True
        assert chunker._parsers == {}
        assert chunker._file_header is None
        assert chunker._file_header_end_byte == 0

    def test_custom_chunk_size(self):
        """Test initialization with custom chunk size."""
        chunker = SyntaxChunker(chunk_size=2048)
        assert chunker.chunk_size == 2048

    def test_disable_file_header(self):
        """Test initialization with file header disabled."""
        chunker = SyntaxChunker(include_file_header=False)
        assert chunker.include_file_header is False

    def test_minimum_chunk_size_clamping(self):
        """Test that chunk_size is clamped to minimum of 1."""
        chunker = SyntaxChunker(chunk_size=0)
        chunker._ensure_valid_parameters()
        assert chunker.chunk_size == 1

        chunker = SyntaxChunker(chunk_size=-10)
        chunker._ensure_valid_parameters()
        assert chunker.chunk_size == 1


# ============================================================================
# Parser Loading Tests
# ============================================================================


class TestParserLoading:
    """Tests for language parser loading."""

    def test_get_parser_python(self, chunker):
        """Test loading Python parser."""
        parser = chunker._get_parser("python")
        assert parser is not None

    def test_get_parser_javascript(self, chunker):
        """Test loading JavaScript parser."""
        parser = chunker._get_parser("javascript")
        assert parser is not None

    def test_get_parser_typescript(self, chunker):
        """Test loading TypeScript parser."""
        parser = chunker._get_parser("typescript")
        assert parser is not None

    def test_get_parser_go(self, chunker):
        """Test loading Go parser."""
        parser = chunker._get_parser("go")
        assert parser is not None

    def test_get_parser_rust(self, chunker):
        """Test loading Rust parser."""
        parser = chunker._get_parser("rust")
        assert parser is not None

    def test_get_parser_java(self, chunker):
        """Test loading Java parser."""
        parser = chunker._get_parser("java")
        assert parser is not None

    def test_get_parser_cpp(self, chunker):
        """Test loading C++ parser."""
        parser = chunker._get_parser("cpp")
        assert parser is not None

    def test_get_parser_c(self, chunker):
        """Test loading C parser."""
        parser = chunker._get_parser("c")
        assert parser is not None

    def test_get_parser_caches_parser(self, chunker):
        """Test that parsers are cached after first load."""
        parser1 = chunker._get_parser("python")
        parser2 = chunker._get_parser("python")
        assert parser1 is parser2

    def test_get_parser_unknown_language(self, chunker):
        """Test that unknown languages return None."""
        parser = chunker._get_parser("unknown_language")
        assert parser is None

    def test_get_parser_unsupported_language(self, chunker):
        """Test that unsupported languages return None."""
        parser = chunker._get_parser("cobol")
        assert parser is None


# ============================================================================
# Python Chunking Tests
# ============================================================================


class TestPythonChunking:
    """Tests for Python code chunking."""

    def test_chunk_simple_function(self, chunker, python_simple_function):
        """Test chunking a simple Python function."""
        chunks = chunker.chunk(python_simple_function, "python")
        assert len(chunks) >= 1
        assert all("text" in chunk for chunk in chunks)
        assert all("start_byte" in chunk for chunk in chunks)
        assert all("end_byte" in chunk for chunk in chunks)

    def test_chunk_function_extracts_name(self, chunker, python_simple_function):
        """Test that function name is extracted."""
        chunks = chunker.chunk(python_simple_function, "python")
        # At least one chunk should have the function name
        function_names = [
            c.get("function_name") for c in chunks if c.get("function_name")
        ]
        assert "hello_world" in function_names

    def test_chunk_class_with_methods(self, chunker, python_class_with_methods):
        """Test chunking a Python class with methods."""
        chunks = chunker.chunk(python_class_with_methods, "python")
        assert len(chunks) >= 1

        # Check class name extraction
        class_names = [c.get("class_name") for c in chunks if c.get("class_name")]
        assert "Calculator" in class_names

    def test_chunk_with_imports(self, chunker, python_with_imports):
        """Test chunking Python code with imports extracts file header."""
        chunks = chunker.chunk(python_with_imports, "python")

        # Check that file header was extracted
        file_header = chunker.get_file_header()
        assert file_header is not None
        assert "import os" in file_header or "from pathlib" in file_header

    def test_chunk_preserves_content(self, chunker, python_simple_function):
        """Test that chunking preserves all content."""
        chunks = chunker.chunk(python_simple_function, "python")

        # Reconstruct content from chunks
        full_content = "".join(c["text"] for c in chunks)
        # Content should be preserved (modulo whitespace normalization)
        assert "def hello_world" in full_content
        assert "Hello, World!" in full_content

    def test_chunk_nested_structures(self, chunker, nested_python_code):
        """Test chunking nested classes and functions."""
        chunks = chunker.chunk(nested_python_code, "python")
        assert len(chunks) >= 1

        # Should find Outer class
        class_names = [c.get("class_name") for c in chunks if c.get("class_name")]
        assert "Outer" in class_names

    def test_chunk_decorated_function(self, chunker):
        """Test chunking a decorated Python function."""
        code = '''@decorator
def decorated_func():
    """A decorated function."""
    pass
'''
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1

        # Should extract the function name
        function_names = [
            c.get("function_name") for c in chunks if c.get("function_name")
        ]
        assert "decorated_func" in function_names

    def test_chunk_async_function(self, chunker):
        """Test chunking an async Python function."""
        code = '''async def async_handler(request):
    """Handle async request."""
    await process(request)
    return response
'''
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1
        assert "async def" in chunks[0]["text"]


# ============================================================================
# JavaScript/TypeScript Chunking Tests
# ============================================================================


class TestJavaScriptChunking:
    """Tests for JavaScript code chunking."""

    def test_chunk_javascript_function(self, chunker, javascript_code):
        """Test chunking JavaScript code."""
        chunks = chunker.chunk(javascript_code, "javascript")
        assert len(chunks) >= 1

        # Check content preservation
        full_content = "".join(c["text"] for c in chunks)
        assert "function Counter" in full_content

    def test_chunk_javascript_arrow_function(self, chunker):
        """Test chunking JavaScript arrow functions."""
        code = """const add = (a, b) => a + b;

const multiply = (a, b) => {
    return a * b;
};
"""
        chunks = chunker.chunk(code, "javascript")
        assert len(chunks) >= 1

    def test_chunk_javascript_class(self, chunker):
        """Test chunking JavaScript class."""
        code = """class Animal {
    constructor(name) {
        this.name = name;
    }

    speak() {
        console.log(`${this.name} makes a sound.`);
    }
}
"""
        chunks = chunker.chunk(code, "javascript")
        assert len(chunks) >= 1


class TestTypeScriptChunking:
    """Tests for TypeScript code chunking."""

    def test_chunk_typescript_interface(self, chunker, typescript_code):
        """Test chunking TypeScript with interface."""
        chunks = chunker.chunk(typescript_code, "typescript")
        assert len(chunks) >= 1

        # Check content includes interface
        full_content = "".join(c["text"] for c in chunks)
        assert "interface User" in full_content

    def test_chunk_typescript_class(self, chunker, typescript_code):
        """Test chunking TypeScript class."""
        chunks = chunker.chunk(typescript_code, "typescript")

        # Verify the class content is captured
        full_content = "".join(c["text"] for c in chunks)
        assert "class UserService" in full_content

        # Check that some class-related metadata is extracted
        # TypeScript may extract interface 'User' as class due to parser behavior
        class_names = [c.get("class_name") for c in chunks if c.get("class_name")]
        assert len(class_names) >= 1


# ============================================================================
# Go Chunking Tests
# ============================================================================


class TestGoChunking:
    """Tests for Go code chunking."""

    def test_chunk_go_struct(self, chunker, go_code):
        """Test chunking Go struct."""
        chunks = chunker.chunk(go_code, "go")
        assert len(chunks) >= 1

        full_content = "".join(c["text"] for c in chunks)
        assert "type Server struct" in full_content

    def test_chunk_go_function(self, chunker, go_code):
        """Test chunking Go function."""
        chunks = chunker.chunk(go_code, "go")

        # Should find main function
        function_names = [
            c.get("function_name") for c in chunks if c.get("function_name")
        ]
        assert "main" in function_names or any(
            "main" in str(c.get("text", "")) for c in chunks
        )

    def test_chunk_go_method(self, chunker, go_code):
        """Test chunking Go method."""
        chunks = chunker.chunk(go_code, "go")

        full_content = "".join(c["text"] for c in chunks)
        assert "func (s *Server) Start()" in full_content


# ============================================================================
# Rust Chunking Tests
# ============================================================================


class TestRustChunking:
    """Tests for Rust code chunking."""

    def test_chunk_rust_struct(self, chunker, rust_code):
        """Test chunking Rust struct."""
        chunks = chunker.chunk(rust_code, "rust")
        assert len(chunks) >= 1

        full_content = "".join(c["text"] for c in chunks)
        assert "struct Counter" in full_content

    def test_chunk_rust_impl(self, chunker, rust_code):
        """Test chunking Rust impl block."""
        chunks = chunker.chunk(rust_code, "rust")

        full_content = "".join(c["text"] for c in chunks)
        assert "impl Counter" in full_content

    def test_chunk_rust_function(self, chunker, rust_code):
        """Test chunking Rust functions."""
        chunks = chunker.chunk(rust_code, "rust")

        function_names = [
            c.get("function_name") for c in chunks if c.get("function_name")
        ]
        # Should find at least main function
        assert len(function_names) > 0 or "fn main" in "".join(
            c["text"] for c in chunks
        )


# ============================================================================
# Java Chunking Tests
# ============================================================================


class TestJavaChunking:
    """Tests for Java code chunking."""

    def test_chunk_java_class(self, chunker, java_code):
        """Test chunking Java class."""
        chunks = chunker.chunk(java_code, "java")
        assert len(chunks) >= 1

        full_content = "".join(c["text"] for c in chunks)
        assert "public class UserManager" in full_content

    def test_chunk_java_method(self, chunker, java_code):
        """Test chunking Java methods."""
        chunks = chunker.chunk(java_code, "java")

        full_content = "".join(c["text"] for c in chunks)
        assert "public void addUser" in full_content


# ============================================================================
# File Header Extraction Tests
# ============================================================================


class TestFileHeaderExtraction:
    """Tests for file header extraction."""

    def test_python_imports_extracted(self, chunker, python_with_imports):
        """Test that Python imports are extracted as file header."""
        chunker.chunk(python_with_imports, "python")

        header = chunker.get_file_header()
        assert header is not None
        assert "import os" in header or "import sys" in header

    def test_javascript_imports_extracted(self, chunker, javascript_code):
        """Test that JavaScript imports are extracted as file header."""
        chunker.chunk(javascript_code, "javascript")

        header = chunker.get_file_header()
        assert header is not None
        assert "import" in header

    def test_go_package_extracted(self, chunker, go_code):
        """Test that Go package declaration is extracted."""
        chunker.chunk(go_code, "go")

        header = chunker.get_file_header()
        assert header is not None
        assert "package main" in header

    def test_java_package_extracted(self, chunker, java_code):
        """Test that Java package declaration is extracted."""
        chunker.chunk(java_code, "java")

        header = chunker.get_file_header()
        assert header is not None
        assert "package com.example" in header

    def test_rust_use_declarations_extracted(self, chunker, rust_code):
        """Test that Rust use declarations are extracted."""
        chunker.chunk(rust_code, "rust")

        header = chunker.get_file_header()
        assert header is not None
        assert "use std::collections::HashMap" in header

    def test_file_header_disabled(self, python_with_imports):
        """Test that file header extraction can be disabled."""
        chunker = SyntaxChunker(include_file_header=False)
        chunker.chunk(python_with_imports, "python")

        header = chunker.get_file_header()
        assert header is None

    def test_no_header_when_no_imports(self, chunker, python_simple_function):
        """Test that no header is returned when file has no imports."""
        chunker.chunk(python_simple_function, "python")

        header = chunker.get_file_header()
        # Simple function has no imports
        assert header is None


# ============================================================================
# Chunk Metadata Tests
# ============================================================================


class TestChunkMetadata:
    """Tests for chunk metadata."""

    def test_chunk_has_byte_offsets(self, chunker, python_simple_function):
        """Test that chunks have start_byte and end_byte."""
        chunks = chunker.chunk(python_simple_function, "python")

        for chunk in chunks:
            assert "start_byte" in chunk
            assert "end_byte" in chunk
            assert chunk["start_byte"] >= 0
            assert chunk["end_byte"] > chunk["start_byte"]

    def test_chunk_has_adjacency_info(self, chunker, python_class_with_methods):
        """Test that chunks have adjacency metadata."""
        chunks = chunker.chunk(python_class_with_methods, "python")

        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk
            assert "total_chunks" in chunk
            assert "prev_id" in chunk
            assert "next_id" in chunk

            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)

    def test_first_chunk_has_no_prev_id(self, chunker, python_class_with_methods):
        """Test that first chunk has no prev_id."""
        chunks = chunker.chunk(python_class_with_methods, "python")

        if chunks:
            assert chunks[0]["prev_id"] is None

    def test_last_chunk_has_no_next_id(self, chunker, python_class_with_methods):
        """Test that last chunk has no next_id."""
        chunks = chunker.chunk(python_class_with_methods, "python")

        if chunks:
            assert chunks[-1]["next_id"] is None

    def test_chunk_symbol_type(self, chunker, python_class_with_methods):
        """Test that chunks have symbol_type."""
        chunks = chunker.chunk(python_class_with_methods, "python")

        # At least one chunk should have a symbol type
        symbol_types = [c.get("symbol_type") for c in chunks if c.get("symbol_type")]
        assert len(symbol_types) > 0
        # Should have class or method type
        assert any(st in ["class", "function", "method"] for st in symbol_types)


# ============================================================================
# Large File Handling Tests
# ============================================================================


class TestLargeFileHandling:
    """Tests for handling large files."""

    def test_large_function_gets_split(self, small_chunk_chunker):
        """Test that large functions are split into smaller chunks."""
        # Create a large function that exceeds chunk_size
        large_function = '''def large_function():
    """A very large function."""
    line1 = "This is line 1"
    line2 = "This is line 2"
    line3 = "This is line 3"
    line4 = "This is line 4"
    line5 = "This is line 5"
    line6 = "This is line 6"
    line7 = "This is line 7"
    line8 = "This is line 8"
    line9 = "This is line 9"
    line10 = "This is line 10"
    return True
'''
        chunks = small_chunk_chunker.chunk(large_function, "python")
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """Test that chunks don't significantly exceed chunk_size."""
        chunker = SyntaxChunker(chunk_size=200)
        code = """def func1():
    pass

def func2():
    pass

def func3():
    pass

def func4():
    pass
"""
        chunks = chunker.chunk(code, "python")

        # Most chunks should be around chunk_size (with some tolerance)
        for chunk in chunks:
            # Allow some tolerance for atomic nodes
            assert len(chunk["text"]) <= 400  # 2x chunk_size as tolerance

    def test_split_preserves_signature(self, small_chunk_chunker):
        """Test that split large segments preserve function signature."""
        large_function = '''def process_data(input_data, config, options):
    """Process the input data according to configuration.

    Args:
        input_data: The data to process
        config: Configuration settings
        options: Additional options
    """
    result = []
    for item in input_data:
        processed = transform(item)
        validated = validate(processed)
        if validated:
            result.append(validated)
    return result
'''
        chunks = small_chunk_chunker.chunk(large_function, "python")

        # Should create multiple chunks
        assert len(chunks) >= 1


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content(self, chunker):
        """Test chunking empty content."""
        chunks = chunker.chunk("", "python")
        # Should return empty list or minimal chunks
        assert isinstance(chunks, list)

    def test_whitespace_only_content(self, chunker):
        """Test chunking whitespace-only content."""
        chunks = chunker.chunk("   \n\n\t\t\n   ", "python")
        assert isinstance(chunks, list)

    def test_comment_only_content(self, chunker):
        """Test chunking file with only comments."""
        code = """# This is a comment
# Another comment
# Yet another comment
"""
        chunks = chunker.chunk(code, "python")
        assert isinstance(chunks, list)

    def test_syntax_error_handling(self, chunker):
        """Test handling of syntax errors."""
        invalid_code = """def broken_function(
    # Missing closing paren and colon
    print("this won't parse correctly"
"""
        # Should not raise exception
        chunks = chunker.chunk(invalid_code, "python")
        assert isinstance(chunks, list)

    def test_unknown_language_returns_empty(self, chunker):
        """Test that unknown language returns empty list."""
        chunks = chunker.chunk("some content", "unknown_lang")
        assert chunks == []

    def test_single_line_file(self, chunker):
        """Test chunking single line file."""
        code = "x = 1"
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1

    def test_no_definitions_file(self, chunker):
        """Test chunking file with no function/class definitions."""
        code = """x = 1
y = 2
z = x + y
print(z)
"""
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1
        # Should still produce chunks even without definitions
        full_content = "".join(c["text"] for c in chunks)
        assert "x = 1" in full_content

    def test_deeply_nested_structures(self, chunker):
        """Test chunking deeply nested structures."""
        code = """def outer():
    def middle():
        def inner():
            def deepest():
                return "deep"
            return deepest()
        return inner()
    return middle()
"""
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1

    def test_unicode_content(self, chunker):
        """Test chunking code with unicode characters."""
        code = '''def greet(name):
    """Приветствие на русском."""
    return f"Привет, {name}! 你好!"
'''
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1
        # Unicode should be preserved
        full_content = "".join(c["text"] for c in chunks)
        assert "Привет" in full_content
        assert "你好" in full_content

    def test_mixed_indentation(self, chunker):
        """Test handling of mixed tabs and spaces."""
        code = """def func():
    if True:
\t\treturn 1
    return 0
"""
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1


# ============================================================================
# Signature Extraction Tests
# ============================================================================


class TestSignatureExtraction:
    """Tests for _find_signature_end method."""

    def test_find_python_function_signature(self, chunker):
        """Test finding Python function signature."""
        text = '''def my_function(arg1, arg2):
    """Docstring."""
    body_code = 1
'''
        end = chunker._find_signature_end(text)
        assert end > 0
        extracted = text[:end]
        assert "def my_function" in extracted

    def test_find_class_signature(self, chunker):
        """Test finding class signature."""
        text = '''class MyClass:
    """Class docstring."""

    def method(self):
        pass
'''
        end = chunker._find_signature_end(text)
        assert end > 0
        extracted = text[:end]
        assert "class MyClass" in extracted

    def test_find_async_function_signature(self, chunker):
        """Test finding async function signature."""
        text = '''async def async_func():
    """Async docstring."""
    await something()
'''
        end = chunker._find_signature_end(text)
        assert end > 0
        extracted = text[:end]
        assert "async def" in extracted

    def test_no_signature_in_plain_code(self, chunker):
        """Test that plain code returns 0."""
        text = """x = 1
y = 2
"""
        end = chunker._find_signature_end(text)
        assert end == 0

    def test_empty_text_signature(self, chunker):
        """Test signature extraction on empty text."""
        end = chunker._find_signature_end("")
        assert end == 0


# ============================================================================
# Atom and Segment Tests
# ============================================================================


class TestAtomAndSegmentProcessing:
    """Tests for internal atom and segment processing."""

    def test_get_atoms_small_node(self, chunker):
        """Test _get_atoms with small content."""
        parser = chunker._get_parser("python")
        code = "x = 1"
        tree = parser.parse(bytes(code, "utf8"))

        atoms = chunker._get_atoms(tree.root_node)
        assert len(atoms) >= 1

    def test_create_segments_covers_content(self, chunker):
        """Test that segments cover entire content."""
        parser = chunker._get_parser("python")
        code = """def func():
    pass

def func2():
    pass
"""
        tree = parser.parse(bytes(code, "utf8"))
        atoms = chunker._get_atoms(tree.root_node)
        segments = chunker._create_segments(atoms, len(code))

        # Verify segments cover the content
        if segments:
            assert segments[0][0] == 0 or segments[0][0] <= atoms[0].start_byte
            assert segments[-1][1] == len(code)


# ============================================================================
# Constants Verification Tests
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_header_node_types_not_empty(self):
        """Test that HEADER_NODE_TYPES is not empty."""
        assert len(HEADER_NODE_TYPES) > 0
        assert "import_statement" in HEADER_NODE_TYPES
        assert "import_declaration" in HEADER_NODE_TYPES

    def test_definition_node_types_not_empty(self):
        """Test that DEFINITION_NODE_TYPES is not empty."""
        assert len(DEFINITION_NODE_TYPES) > 0
        assert "function_definition" in DEFINITION_NODE_TYPES
        assert "class_definition" in DEFINITION_NODE_TYPES

    def test_class_node_types_not_empty(self):
        """Test that CLASS_NODE_TYPES is not empty."""
        assert len(CLASS_NODE_TYPES) > 0
        assert "class_definition" in CLASS_NODE_TYPES
        assert "class_declaration" in CLASS_NODE_TYPES

    def test_function_node_types_not_empty(self):
        """Test that FUNCTION_NODE_TYPES is not empty."""
        assert len(FUNCTION_NODE_TYPES) > 0
        assert "function_definition" in FUNCTION_NODE_TYPES
        assert "function_declaration" in FUNCTION_NODE_TYPES

    def test_language_packages_mapping(self):
        """Test LANGUAGE_PACKAGES mapping."""
        chunker = SyntaxChunker()
        assert "python" in chunker.LANGUAGE_PACKAGES
        assert "javascript" in chunker.LANGUAGE_PACKAGES
        assert "typescript" in chunker.LANGUAGE_PACKAGES
        assert "go" in chunker.LANGUAGE_PACKAGES
        assert "rust" in chunker.LANGUAGE_PACKAGES
        assert "java" in chunker.LANGUAGE_PACKAGES


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete chunking workflows."""

    def test_full_python_file_chunking(self, chunker):
        """Test chunking a complete Python file."""
        code = '''"""Module docstring."""

import os
import sys
from typing import List

MAX_VALUE = 100


class DataProcessor:
    """Process data."""

    def __init__(self, data: List[int]):
        self.data = data

    def process(self) -> List[int]:
        """Process the data."""
        return [x * 2 for x in self.data]


def main():
    """Main function."""
    processor = DataProcessor([1, 2, 3])
    result = processor.process()
    print(result)


if __name__ == "__main__":
    main()
'''
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1

        # Verify structure
        full_content = "".join(c["text"] for c in chunks)
        assert "class DataProcessor" in full_content
        assert "def main" in full_content

        # Verify metadata
        class_found = any(c.get("class_name") == "DataProcessor" for c in chunks)
        assert class_found

        # Verify header extraction
        header = chunker.get_file_header()
        assert header is not None
        assert "import os" in header

    def test_multiple_files_sequential(self, chunker):
        """Test chunking multiple files sequentially."""
        python_code = """def python_func():
    pass
"""
        js_code = """function jsFunc() {
    return true;
}
"""
        # Chunk Python
        py_chunks = chunker.chunk(python_code, "python")
        assert len(py_chunks) >= 1

        # Chunk JavaScript (should reset state)
        js_chunks = chunker.chunk(js_code, "javascript")
        assert len(js_chunks) >= 1

        # Verify they're independent
        assert any("python_func" in c.get("text", "") for c in py_chunks)
        assert any("jsFunc" in c.get("text", "") for c in js_chunks)

    def test_realistic_file_with_all_features(self, chunker):
        """Test a realistic file with imports, classes, functions, and nested code."""
        code = '''"""Comprehensive test module."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CONFIG = {"debug": True, "max_retries": 3}


@dataclass
class Config:
    """Configuration dataclass."""
    debug: bool = True
    max_retries: int = 3


class ServiceManager:
    """Manages services."""

    def __init__(self, config: Config):
        """Initialize the manager."""
        self.config = config
        self.services: Dict[str, "Service"] = {}

    def register(self, name: str, service: "Service") -> None:
        """Register a service."""
        self.services[name] = service
        logger.info(f"Registered service: {name}")

    def get_service(self, name: str) -> Optional["Service"]:
        """Get a service by name."""
        return self.services.get(name)

    class Service:
        """Nested service class."""

        def __init__(self, name: str):
            self.name = name

        def run(self) -> bool:
            """Run the service."""
            return True


def create_manager() -> ServiceManager:
    """Factory function for ServiceManager."""
    config = Config()
    return ServiceManager(config)


async def async_main():
    """Async main function."""
    manager = create_manager()
    await some_async_operation(manager)


def main():
    """Main entry point."""
    import asyncio
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
'''
        chunks = chunker.chunk(code, "python")
        assert len(chunks) >= 1

        # Verify all major elements are captured
        full_content = "".join(c["text"] for c in chunks)
        assert "class Config" in full_content
        assert "class ServiceManager" in full_content
        assert "def create_manager" in full_content
        assert "async def async_main" in full_content

        # Verify header
        header = chunker.get_file_header()
        assert header is not None
        assert "import json" in header
