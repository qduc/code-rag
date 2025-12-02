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
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource

from .api import CodeRAGAPI


# Global API instance
api: Optional[CodeRAGAPI] = None


def format_search_results(
    results: List[Dict[str, Any]],
    show_full_content: bool = False,
    content_limit: int = 600,
) -> str:
    """Format search results as a compact, token-efficient string optimized for AI consumption.

    Args:
        results: Search results from the API
        show_full_content: If True, show full content; if False, truncate long results
        content_limit: Maximum characters to show per result (default: 600)

    Format (minimizes tokens while preserving clarity):
        file_path:start-end | func_name | ClassName (score)
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
        content = result.get("expanded_content", result["content"])

        # Use expanded line numbers if available
        if "expanded_start_line" in result:
            start_line = result["expanded_start_line"]
            end_line = result["expanded_end_line"]

        # Extract symbol info for header
        function_name = result.get("function_name")
        class_name = result.get("class_name")

        # Build header: file:lines | symbol_info (score)
        if start_line and end_line:
            header_parts = [f"{file_path}:{start_line}-{end_line}"]
        else:
            header_parts = [file_path]

        # Add symbol context
        symbol_parts = []
        if function_name:
            symbol_parts.append(f"{function_name}()")
        if class_name:
            symbol_parts.append(class_name)

        if symbol_parts:
            header_parts.append(" | ".join(symbol_parts))

        header_parts.append(f"({similarity:.2f})")
        header = " ".join(header_parts)

        output_lines.append(header)

        # Content (truncate if needed)
        if show_full_content:
            output_lines.append(content)
        else:
            # Truncate to content_limit chars
            if len(content) > content_limit:
                display_content = content[:content_limit] + "…"
            else:
                display_content = content
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
                "Semantic code search using natural language queries.\n"
                "\n"
                "Best for exploratory searches when you don't know exact file names or method names.\n"
                "\n"
                "Ideal queries:\n"
                "  • 'where are API keys validated' → finds validation logic regardless of variable names\n"
                "  • 'code that converts markdown to HTML' → finds converters even if not named 'markdown'\n"
                "  • 'retry logic for failed requests' → finds retry patterns across different implementations\n"
                "  • 'caching strategy for expensive operations' → discovers cache implementations by behavior\n"
                "\n"
                "Returns: file:lines | function() | ClassName (score) + code snippet.\n"
                "Scoring: Cross-encoder logits (unbounded, typically -10 to +10). Higher = more relevant.\n"
                "Compare scores relatively within each query; magnitude matters more than sign."
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
                    "expand_context": {
                        "type": "boolean",
                        "description": "If true, include surrounding code for more context (slightly slower)",
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
            expand_context = arguments.get("expand_context", False)

            if not codebase_path:
                return [TextContent(type="text", text="Error: 'codebase_path' is required")]
            if not query:
                return [TextContent(type="text", text="Error: 'query' is required")]

            # Auto-index if needed using unified method
            # MCP mode: no validation, no user confirmation, silent indexing
            # Collection name is auto-generated from codebase path for uniqueness
            result = api.ensure_indexed(
                codebase_path,
                collection_name=None,  # Auto-generate unique name per codebase
                force_reindex=False,
                validate_codebase=False,  # Skip validation in MCP (auto-accept)
                validation_callback=None,
                progress_callback=None,  # Silent progress
            )

            if not result["success"]:
                return [TextContent(type="text", text=result.get("error", "Unknown error"))]

            # Perform search with optional context expansion
            # Note: search will use the same auto-generated collection name
            results = api.search(
                query,
                n_results=max_results,
                expand_context=expand_context,
            )

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


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) for clean exit."""
    print("\nShutting down Code-RAG MCP server...", file=sys.stderr)
    sys.exit(0)


def main():
    """Entry point for the MCP server (synchronous wrapper)."""
    # Register signal handler for clean exit on Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        # This handles the case where asyncio.run itself is interrupted
        print("\nShutting down Code-RAG MCP server...", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
