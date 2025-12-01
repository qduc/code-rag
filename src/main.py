"""Main CLI entry point for code-rag tool."""

import argparse
import sys
from pathlib import Path


def main():
    """Main entry point for the code-rag CLI tool."""
    parser = argparse.ArgumentParser(
        description="Convert a codebase into vector embeddings for searching and analysis."
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None,
    )

    parser.add_argument(
        "--database",
        type=str,
        help="Type of vector database to use (default: chroma)",
        default="chroma",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to use",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path to the codebase root directory",
        default=".",
    )

    args = parser.parse_args()

    # TODO: Implement CLI logic
    # 1. Check if codebase has been processed
    # 2. If not, start processing
    # 3. Start query session

    print("Code-RAG CLI initialized")
    print(f"Codebase path: {args.path}")
    print(f"Database type: {args.database}")
    print(f"Embedding model: {args.model}")


if __name__ == "__main__":
    main()
