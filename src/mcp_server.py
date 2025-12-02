"""MCP server for Code-RAG.

This module provides an MCP (Model Context Protocol) server that exposes
Code-RAG functionality as tools that Claude can invoke.

Design Philosophy:
- Auto-indexing: Automatically index codebases on first search
- Transparent: Hide implementation details (chunking, collections, etc.)
- Simple: Focused tools that "just work"
- Context-aware: Provide enough context to reduce follow-up reads
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .api import CodeRAGAPI


# Global API instance
api: Optional[CodeRAGAPI] = None


def read_file_content(
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read file content, optionally for a specific line range.

    Args:
        file_path: Path to the file to read
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (1-indexed, inclusive)

    Returns:
        File content as a string, with line numbers prefixed
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1 for binary-ish files
        content = path.read_text(encoding="latin-1")

    lines = content.splitlines()
    total_lines = len(lines)

    # Determine line range
    if start_line is None:
        start_line = 1
    if end_line is None:
        end_line = total_lines

    # Validate and clamp line numbers
    start_line = max(1, min(start_line, total_lines))
    end_line = max(start_line, min(end_line, total_lines))

    # Extract the requested range (convert to 0-indexed)
    selected_lines = lines[start_line - 1 : end_line]

    # Format with line numbers
    output_lines = []
    for i, line in enumerate(selected_lines, start=start_line):
        output_lines.append(f"{i:4d}. {line}")

    return "\n".join(output_lines)


def format_search_results(results: List[Dict[str, Any]], show_full_content: bool = False) -> str:
    """Format search results as a readable text string.

    Args:
        results: Search results from the API
        show_full_content: If True, show full content; if False, truncate long results
    """
    if not results:
        return "No results found."

    output_lines = [f"Found {len(results)} relevant code locations:\n"]

    for i, result in enumerate(results):
        file_path = result["file_path"]
        start_line = result.get("start_line")
        end_line = result.get("end_line")
        similarity = result["similarity"]
        content = result["content"]

        output_lines.append("-" * 60)
        output_lines.append(f"[{i + 1}] {file_path}")

        # Show line numbers if available (cleaner than chunk index)
        if start_line and end_line:
            output_lines.append(f"    Lines {start_line}-{end_line} | Relevance: {similarity:.1%}")
        else:
            output_lines.append(f"    Relevance: {similarity:.1%}")

        output_lines.append("-" * 60)

        # Truncate or show full content based on parameter
        if show_full_content:
            output_lines.append(content)
        else:
            # Smart truncation: show up to 400 chars
            display_content = content[:400] + "..." if len(content) > 400 else content
            output_lines.append(display_content)
        output_lines.append("")

    return "\n".join(output_lines)


# Create the MCP server
server = Server("code-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Code-RAG tools.

    Philosophy:
    - search_codebase: Semantic search for discovering functionality
    - get_file_content: Read specific files or line ranges for deep context
    """
    return [
        Tool(
            name="search_codebase",
            description=(
                "Search for code in a codebase using natural language queries. "
                "Automatically indexes the codebase on first use. "
                "\n\n"
                "BEST FOR:\n"
                "- Discovering where functionality is implemented\n"
                "- Finding code patterns or examples you don't know the location of\n"
                "- Understanding how features work across the codebase\n"
                "- Exploratory searches ('how does X work?')\n"
                "\n"
                "Example queries:\n"
                "- 'authentication and login logic'\n"
                "- 'database connection setup'\n"
                "- 'error handling for API requests'\n"
                "\n"
                "Returns: Relevant code snippets with file paths and line numbers.\n"
                "TIP: Use show_full_content=true for more context, or use "
                "get_file_content to read full files after finding them."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "codebase_path": {
                        "type": "string",
                        "description": "Absolute path to the codebase root directory (e.g., '/home/user/myproject')",
                    },
                    "query": {
                        "type": "string",
                        "description": "What you're looking for, in natural language (e.g., 'authentication logic')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5, max: 20)",
                        "default": 5,
                    },
                    "show_full_content": {
                        "type": "boolean",
                        "description": "If true, show full chunk content without truncation (default: false)",
                        "default": False,
                    },
                },
                "required": ["codebase_path", "query"],
            },
        ),
        Tool(
            name="get_file_content",
            description=(
                "Read the content of a specific file, optionally with a line range. "
                "Use this after search_codebase to get full context around a match.\n\n"
                "BEST FOR:\n"
                "- Reading full file content after finding it via search\n"
                "- Getting more context around a specific line range\n"
                "- Viewing implementation details of a known file\n"
                "\n"
                "Returns: File content with line numbers.\n"
                "TIP: When you know the exact file path, this is faster than search."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed, optional)",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-indexed, inclusive, optional)",
                    },
                },
                "required": ["file_path"],
            },
        ),
    ]


# Removed: _ensure_indexed is now unified in api.ensure_indexed()


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls from Claude.

    Philosophy: Keep it simple and transparent. Auto-index when needed.
    """
    global api

    if name == "get_file_content":
        # Handle get_file_content separately - doesn't need API
        file_path = arguments.get("file_path")
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")

        if not file_path:
            return [TextContent(type="text", text="Error: 'file_path' is required")]

        try:
            content = read_file_content(file_path, start_line, end_line)
            # Add header with file info
            path = Path(file_path)
            header = f"File: {file_path}\n"
            if start_line or end_line:
                header += f"Lines: {start_line or 1}-{end_line or 'end'}\n"
            header += "-" * 60 + "\n"
            return [TextContent(type="text", text=header + content)]
        except FileNotFoundError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error reading file: {e}")]

    if api is None:
        return [
            TextContent(
                type="text",
                text="Error: Code-RAG API not initialized. Server startup may have failed.",
            )
        ]

    try:
        if name == "search_codebase":
            codebase_path = arguments.get("codebase_path")
            query = arguments.get("query")
            max_results = min(arguments.get("max_results", 5), 20)  # Cap at 20
            show_full_content = arguments.get("show_full_content", False)

            if not codebase_path:
                return [TextContent(type="text", text="Error: 'codebase_path' is required")]
            if not query:
                return [TextContent(type="text", text="Error: 'query' is required")]

            # Auto-index if needed using unified method
            # MCP mode: no validation, no user confirmation, silent indexing
            result = api.ensure_indexed(
                codebase_path,
                collection_name="codebase",
                force_reindex=False,
                validate_codebase=False,  # Skip validation in MCP (auto-accept)
                validation_callback=None,
                progress_callback=None,  # Silent progress
            )

            if not result["success"]:
                return [TextContent(type="text", text=result.get("error", "Unknown error"))]

            # Perform search
            results = api.search(query, n_results=max_results, collection_name="codebase")

            # Format and return results
            formatted = format_search_results(results, show_full_content=show_full_content)
            return [TextContent(type="text", text=formatted)]

        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return [TextContent(type="text", text=f"Error executing '{name}':\n{error_details}")]


async def async_main():
    """Run the MCP server (async implementation)."""
    global api

    # Initialize the Code-RAG API
    # These can be customized via environment variables (CODE_RAG_*)
    try:
        api = CodeRAGAPI(
            database_type="chroma",  # Default to ChromaDB
            reranker_enabled=True,  # Enable reranking by default
        )
        print("Code-RAG MCP server initialized", file=sys.stderr)
    except Exception as e:
        print(f"Failed to initialize Code-RAG API: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    """Entry point for the MCP server (synchronous wrapper)."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
