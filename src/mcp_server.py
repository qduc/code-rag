"""MCP server for Code-RAG.

This module provides an MCP (Model Context Protocol) server that exposes
Code-RAG functionality as tools that Claude can invoke.

Design Philosophy:
- Auto-indexing: Automatically index codebases on first search
- Transparent: Hide implementation details (chunking, collections, etc.)
- Simple: One main tool that "just works"
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

    Philosophy: One tool, does one thing well - semantic code search.
    Everything else (reading files, etc.) Claude already has tools for.
    """
    return [
        Tool(
            name="search_codebase",
            description=(
                "Search for code in a codebase using natural language queries. "
                "Automatically indexes the codebase on first use (transparent to you). "
                "\n\n"
                "Use this when you need to:\n"
                "- Find where specific functionality is implemented\n"
                "- Locate code patterns or examples\n"
                "- Understand how a feature works\n"
                "- Search for error handling, API calls, database logic, etc.\n"
                "\n"
                "Example queries:\n"
                "- 'authentication and login logic'\n"
                "- 'database connection setup'\n"
                "- 'error handling for API requests'\n"
                "- 'user registration implementation'\n"
                "- 'how files are uploaded'\n"
                "\n"
                "Returns: Relevant code snippets with file paths and line numbers. "
                "You can then use the Read tool to see full file contents if needed."
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
                        "description": "If true, show full code snippets without truncation for more context (default: false)",
                        "default": False,
                    },
                },
                "required": ["codebase_path", "query"],
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
