"""MCP server for Code-RAG.

This module provides an MCP (Model Context Protocol) server that exposes
Code-RAG functionality as tools that Claude can invoke.

Design Philosophy:
- Auto-indexing: Automatically index codebases on first search
- Transparent: Hide implementation details (chunking, collections, etc.)
- Simple: One main tool that "just works"
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .api import CodeRAGAPI


# Global API instance
api: Optional[CodeRAGAPI] = None


def format_search_results(results: List[Dict[str, Any]], show_full_content: bool = False) -> str:
    """Format search results as a compact, token-efficient string optimized for AI consumption.

    Args:
        results: Search results from the API
        show_full_content: If True, show full content; if False, truncate long results

    Format (minimizes tokens while preserving clarity):
        file_path:start-end (score)
        <content>
        ---
    """
    if not results:
        return "No results."

    output_lines = []

    for result in results:
        file_path = result["file_path"]
        start_line = result.get("start_line")
        end_line = result.get("end_line")
        similarity = result["similarity"]
        content = result["content"]

        # Compact header: file:lines (score)
        if start_line and end_line:
            header = f"{file_path}:{start_line}-{end_line} ({similarity:.2f})"
        else:
            header = f"{file_path} ({similarity:.2f})"

        output_lines.append(header)

        # Content (truncate if needed)
        if show_full_content:
            output_lines.append(content)
        else:
            # Truncate to 300 chars (reduced from 400 to save tokens)
            display_content = content[:300] + "…" if len(content) > 300 else content
            output_lines.append(display_content)

        # Minimal separator
        output_lines.append("---")

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
                "Search code using natural language queries. Auto-indexes on first use. "
                "Returns compact results: file:lines (score) + code snippet.\n"
                "\n"
                "Use for: finding implementations, locating patterns, understanding features, "
                "searching error handling, APIs, database logic.\n"
                "\n"
                "Query examples: 'authentication logic', 'database setup', 'error handling', "
                "'file upload', 'user registration'.\n"
                "\n"
                "Returns token-efficient format. Use Read tool for full file contents."
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
            formatted = format_search_results(results, show_full_content=False)
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
