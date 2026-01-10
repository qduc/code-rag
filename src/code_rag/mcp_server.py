"""MCP server for Code-RAG.

This module provides an MCP (Model Context Protocol) server that exposes
Code-RAG functionality as tools that Claude can invoke.

Design Philosophy:
- Auto-indexing: Automatically index codebases on first search
- Transparent: Hide implementation details (chunking, collections, etc.)
- Simple: One main tool that "just works"
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from .config.config import Config

# from dotenv import load_dotenv
# env_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(env_path)

# Global API instance
api: Optional[Any] = None

# Event set when `api` has been initialized in background thread
api_ready_event = threading.Event()


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
                "PRIMARY TOOL for finding code in a codebase using semantic search.\n"
                "  Query Strategy (Best -> Good -> Avoid):\n"
                "  1. BEST: Specific concepts + domain terminology\n"
                "     'stream retry logic when upstream fails' -> finds retry mechanisms\n"
                "     'client disconnect cleanup during active request' -> finds connection handling\n"
                "     'authentication token refresh before expiry' -> finds token management\n"
                "  2. GOOD: Known identifiers + behavior context\n"
                "     'StreamingNotSupportedError retry logic' -> finds exact error handling\n"
                "     'persistence.markError cleanup' -> finds error state management\n"
                "     'handleRegularStreaming client disconnect' -> finds specific handler\n"
                "  3. EXPLORATORY: Natural behavior descriptions\n"
                "     'code that validates API keys before requests'\n"
                "     'retry logic for failed network requests'\n"
                "     'caching strategy to avoid expensive operations'\n"
                "  4. AVOID: Syntax, generic terms, or open-ended questions\n"
                "     'req.on(\"close\", () => {' -> semantic search doesn't understand syntax\n"
                "     'error handling' -> too broad, use specific error types/scenarios\n"
                "     'how does auth work' -> not a search query, ask for specific components\n"
                "     'explain the architecture' -> search concrete parts: 'request routing', 'middleware pipeline'\n"
                "  Tips:\n"
                "  - Semantic search finds code, not explanations — you synthesize the answer\n"
                "  - More specific concepts = better results (use domain terminology)\n"
                "  - If open-ended, break into specific searches: 'cache invalidation', 'cache TTL', 'cache keys'\n"
                "  - Results ranked by relevance score (higher = more relevant)\n"
                "  - Follow up with Read/Grep for complete understanding"
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
                    "file_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by file extensions (e.g. ['.py', '.md']). Case-insensitive.",
                    },
                    "include_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only include files whose paths contain these strings "
                            "(e.g. ['src/code_rag/api', 'tests/'])."
                        ),
                    },
                },
                "required": ["codebase_path", "query"],
            },
        ),
    ]


# Removed: _ensure_indexed is now unified in api.ensure_indexed()


@server.call_tool()
async def call_tool(
    name: str, arguments: Any, _api_wait_timeout: float = 30.0
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Handle tool calls from Claude.

    Philosophy: Keep it simple and transparent. Auto-index when needed.

    Args:
        name: Tool name to execute
        arguments: Tool arguments
        _api_wait_timeout: Internal parameter for testing - timeout in seconds to wait for API initialization
    """
    # global api

    if api is None:
        # If a request arrives very shortly after server startup, the API may still be
        # initializing in the background thread. Wait briefly for initialization so we
        # don't immediately return an error for a transient race condition.
        # We wait up to 30 seconds to allow for model loading (e.g., downloading embeddings).
        try:
            loop = asyncio.get_running_loop()

            # Wait for the API to be ready in the background thread
            # This accounts for model loading time on first startup
            def wait_for_api(timeout=_api_wait_timeout):
                return api_ready_event.wait(timeout)

            await loop.run_in_executor(None, wait_for_api)
        except Exception:
            # If waiting fails for any reason (no running loop etc.), fall through and return
            pass

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
            file_types = arguments.get("file_types")
            include_paths = arguments.get("include_paths")

            if not codebase_path:
                return [
                    TextContent(type="text", text="Error: 'codebase_path' is required")
                ]
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
                return [
                    TextContent(type="text", text=result.get("error", "Unknown error"))
                ]

            # Perform search with optional context expansion
            # Note: search will use the same auto-generated collection name
            results = api.search(
                query,
                n_results=max_results,
                expand_context=expand_context,
                file_types=file_types,
                include_paths=include_paths,
            )

            # Format and return results
            formatted = format_search_results(results, show_full_content=False)
            return [TextContent(type="text", text=formatted)]

        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    except Exception:
        import traceback

        error_details = traceback.format_exc()
        return [
            TextContent(type="text", text=f"Error executing '{name}':\n{error_details}")
        ]


async def async_main():
    """Run the MCP server (async implementation)."""
    # global api

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        # Start API import and initialization in background thread for fast startup
        import threading

        def preload_api():
            global api
            try:
                from .api import CodeRAGAPI

                config = Config()
                api = CodeRAGAPI(
                    database_type=config.get_database_type(),
                    embedding_model=config.get_embedding_model(),
                    reranker_enabled=config.is_reranker_enabled(),
                    lazy_load_models=True,  # Defer model loading for fast startup
                )
                print(
                    "Code-RAG MCP server initialized (models loading in background)",
                    file=sys.stderr,
                )
                # Start loading models in background immediately after initialization
                api.start_background_loading()
                # Signal that `api` has been initialized and is ready to accept calls
                try:
                    api_ready_event.set()
                except Exception:
                    pass
            except Exception as e:
                print(f"Failed to initialize Code-RAG API: {e}", file=sys.stderr)
                # Don't exit here, let the server run and handle errors in call_tool

        threading.Thread(target=preload_api, daemon=True).start()

        # Create the initialization options and run loop
        init_options = server.create_initialization_options()
        server_task = asyncio.create_task(
            server.run(read_stream, write_stream, init_options)
        )

        # Setup Shutdown Logic
        #
        # Problem: MCP clients (like Claude Code) send SIGINT while keeping stdin open,
        # expecting shutdown within ~100ms. However, stdio_server() blocks waiting for
        # stdin EOF, preventing clean exit.
        #
        # Solution:
        # 1. Close fd 0 to force stdin EOF and unblock stdio_server()
        # 2. Cancel both server_task and main_task to exit the context manager
        # 3. Use os._exit(0) after shutdown to bypass asyncio.run() hanging
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
        shutting_down = False

        def signal_handler(sig: int):
            nonlocal shutting_down
            if shutting_down:
                return
            shutting_down = True

            try:
                sig_name = signal.Signals(sig).name
            except Exception:
                sig_name = str(sig)
            print(f"\nReceived {sig_name}, initiating shutdown...", file=sys.stderr)

            # Close stdin (fd 0) to force EOF and unblock stdio_server().
            # The MCP stdio_server() intentionally does not close stdio handles
            # and will otherwise wait indefinitely for stdin EOF.
            try:
                os.close(0)
            except Exception:
                pass

            # Cancel both tasks to ensure clean context manager exit
            server_task.cancel()
            if main_task is not None:
                main_task.cancel()

        # Register the signal handler
        loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT)
        loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM)

        print("MCP Server running...", file=sys.stderr)

        try:
            await server_task
        except asyncio.CancelledError:
            pass

        print("Shutdown complete.", file=sys.stderr)

        # Force process exit after signal-initiated shutdown.
        # After os.close(0), asyncio.run() may hang waiting for event loop cleanup.
        # Using os._exit(0) ensures immediate termination and meets Claude Code's
        # 100ms shutdown timeout requirement.
        if shutting_down:
            os._exit(0)


def main():
    """Entry point for the MCP server (synchronous wrapper)."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        # Fallback for environments where signal handlers might miss
        pass


if __name__ == "__main__":
    main()
