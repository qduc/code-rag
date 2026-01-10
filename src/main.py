"""Main CLI entry point for code-rag tool."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .api import CodeRAGAPI
from .config.config import Config
from .database.database_interface import DatabaseInterface
from .embeddings.embedding_interface import EmbeddingInterface
from .processor.file_processor import FileProcessor
from .reranker.reranker_interface import RerankerInterface

# env_path = Path(__file__).resolve().parent / ".env"
# load_dotenv(env_path)


def process_codebase(
    root_path: str,
    database: DatabaseInterface,
    embedding_model: EmbeddingInterface,
    config: Config,
) -> int:
    """
    Process the codebase and store embeddings in the database.

    Args:
        root_path: Path to the codebase root directory
        database: Database instance to store embeddings
        embedding_model: Embedding model to generate vectors
        config: Configuration object

    Returns:
        Number of chunks processed
    """
    processor = FileProcessor(
        exclude_tests=config.should_exclude_tests(),
        include_file_header=config.should_include_file_header(),
        additional_ignore_patterns=config.get_additional_ignore_patterns(),
    )

    print("Discovering files...")
    files = processor.discover_files(root_path)
    print(f"Found {len(files)} files to process")

    if not files:
        print("No files found to process.")
        return 0

    total_chunks = 0
    batch_ids = []
    batch_contents = []
    batch_metadatas = []

    chunk_size = config.get_chunk_size()
    batch_size = config.get_batch_size()

    for i, file_path in enumerate(files):
        # Show progress
        rel_path = str(Path(file_path).relative_to(Path(root_path).resolve()))
        print(f"Processing ({i + 1}/{len(files)}): {rel_path}")

        # Process file into chunks
        chunks = processor.process_file(file_path, chunk_size)

        for chunk_data in chunks:
            batch_ids.append(chunk_data["id"])
            batch_contents.append(chunk_data["content"])
            batch_metadatas.append(chunk_data["metadata"])

            # When batch is full, embed and store
            if len(batch_ids) >= batch_size:
                embeddings = embedding_model.embed_batch(batch_contents)
                database.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                )
                total_chunks += len(batch_ids)
                batch_ids = []
                batch_contents = []
                batch_metadatas = []

    # Process remaining batch
    if batch_ids:
        embeddings = embedding_model.embed_batch(batch_contents)
        database.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_contents,
            metadatas=batch_metadatas,
        )
        total_chunks += len(batch_ids)

    return total_chunks


def query_session(
    database: DatabaseInterface,
    embedding_model: EmbeddingInterface,
    reranker: Optional[RerankerInterface] = None,
    n_results: int = 5,
    reranker_multiplier: int = 2,
) -> None:
    """
    Start an interactive query session.

    Args:
        database: Database instance to query
        embedding_model: Embedding model to generate query vectors
        reranker: Optional reranker for result refinement
        n_results: Number of results to return per query
        reranker_multiplier: How many results to retrieve before reranking
    """
    print("\n" + "=" * 60)
    print("Query Session Started")
    print("Enter your query to search the codebase.")
    print("Press Ctrl+C to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("\nQuery: ").strip()

            if not query:
                print("Please enter a non-empty query.")
                continue

            # Generate embedding for query using embed_query to support models with query-specific prefixes
            query_embedding = embedding_model.embed_query(query)

            # Determine how many results to retrieve from database
            if reranker is not None:
                db_n_results = n_results * reranker_multiplier
            else:
                db_n_results = n_results

            # Search database
            results = database.query(query_embedding, n_results=db_n_results)

            # Display results
            if not results["documents"] or not results["documents"][0]:
                print("\nNo results found.")
                continue

            # Apply reranking if enabled
            if reranker is not None:
                try:
                    # Extract documents from results
                    documents = results["documents"][0]

                    # Rerank documents
                    reranked_indices = reranker.rerank(
                        query, documents, top_k=n_results
                    )

                    # Reorder results based on reranking
                    reranked_docs = []
                    reranked_metadata = []
                    reranked_scores = []

                    for orig_idx, rerank_score in reranked_indices:
                        reranked_docs.append(results["documents"][0][orig_idx])
                        reranked_metadata.append(results["metadatas"][0][orig_idx])
                        reranked_scores.append(rerank_score)

                    # Update results with reranked data
                    results["documents"][0] = reranked_docs
                    results["metadatas"][0] = reranked_metadata
                    results["distances"][0] = reranked_scores

                except Exception as e:
                    print(f"\nWarning: Reranking failed ({e}), using original results")

            print(f"\nFound {len(results['documents'][0])} results:\n")

            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                file_path = metadata.get("file_path", "Unknown")
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")
                function_name = metadata.get("function_name")
                class_name = metadata.get("class_name")

                # Calculate similarity score
                if reranker is not None:
                    similarity = distance  # Already a relevance score (higher = better)
                else:
                    similarity = 1 - distance  # Convert cosine distance to similarity

                # Build header: file:lines | symbol_info (score)
                # Format matches MCP server output for consistency
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

                print(header)

                # Truncate long documents for display (600 chars to match MCP default)
                display_doc = doc[:600] + "…" if len(doc) > 600 else doc
                print(display_doc)
                print("---")

        except KeyboardInterrupt:
            print("\n\nExiting query session. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during query: {e}")


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
        choices=["chroma", "qdrant"],
        help="Type of vector database to use (default: chroma)",
        default="chroma",
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Embedding model to use",
        default=None,
    )

    parser.add_argument(
        "--path",
        type=str,
        help="Path to the codebase root directory",
        default=".",
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Force reprocessing of the codebase",
    )

    parser.add_argument(
        "--results",
        type=int,
        help="Number of results to return per query (default: 5)",
        default=5,
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable semantic reranking of results",
    )

    parser.add_argument(
        "--reranker-model",
        type=str,
        help="Cross-encoder model to use for reranking",
        default=None,
    )

    parser.add_argument(
        "--reranker-multiplier",
        type=int,
        help="Retrieval multiplier for reranking (default: 2)",
        default=None,
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config with command line arguments
    if args.model:
        config.embedding_model = args.model

    # Resolve codebase path
    codebase_path = Path(args.path).resolve()
    if not codebase_path.exists():
        print(f"Error: Path '{args.path}' does not exist.")
        sys.exit(1)

    if not codebase_path.is_dir():
        print(f"Error: Path '{args.path}' is not a directory.")
        sys.exit(1)

    print("=" * 60)
    print("Code-RAG: Codebase Vector Search Tool")
    print("=" * 60)
    print(f"Codebase path: {codebase_path}")
    print(f"Database type: {args.database}")
    print(f"Embedding model: {config.embedding_model}")

    # Reranker info will be added after initialization
    reranker_enabled_preview = config.is_reranker_enabled() and not args.no_rerank
    print(f"Reranking: {'enabled' if reranker_enabled_preview else 'disabled'}")

    print("=" * 60)

    # Get configuration
    db_path = config.get_database_path()
    reranker_enabled = config.is_reranker_enabled() and not args.no_rerank
    reranker_model_name = args.reranker_model or config.get_reranker_model()
    reranker_multiplier = args.reranker_multiplier or config.get_reranker_multiplier()

    # Initialize API with configuration
    print("\nInitializing Code-RAG API...")
    api = CodeRAGAPI(
        database_type=args.database,
        database_path=db_path,
        embedding_model=config.embedding_model,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model_name if reranker_enabled else None,
        reranker_multiplier=reranker_multiplier,
    )

    print(f"Embedding model loaded: {api.embedding_model_name}")
    print(f"Embedding dimension: {api.embedding_model.get_embedding_dimension()}")
    if api.reranker:
        print(f"Reranker loaded: {reranker_model_name}")
    else:
        print("Reranking disabled")

    # Define validation callback for user confirmation
    def validation_callback(path: Path) -> bool:
        """Ask user for confirmation if path doesn't look like a codebase."""
        print(
            "\nWarning: The target directory does not appear to contain a typical codebase."
        )
        try:
            resp = input("Continue indexing this directory? [y/N]: ").strip().lower()
            return resp in ("y", "yes")
        except KeyboardInterrupt:
            print("\nAborting.")
            return False

    # Define progress callback for indexing
    def progress_callback(current: int, total: int, file_path: str):
        """Show progress during indexing."""
        rel_path = str(Path(file_path).relative_to(codebase_path))
        print(f"Processing ({current}/{total}): {rel_path}")

    # Use unified ensure_indexed method
    # CLI mode: with validation and user confirmation
    # Collection name is auto-generated from codebase path for uniqueness
    result = api.ensure_indexed(
        str(codebase_path),
        collection_name=None,  # Auto-generate unique name per codebase
        force_reindex=args.reindex,
        validate_codebase=True,  # Enable validation in CLI
        validation_callback=validation_callback,
        progress_callback=progress_callback,
    )

    if not result["success"]:
        print(f"\nError: {result['error']}")
        sys.exit(1)

    # Report results
    if result.get("reloaded_model"):
        print(f"\nReloaded with stored embedding model: {result['reloaded_model']}")

    if result.get("already_indexed"):
        print(f"\nCodebase already processed. Found {result['total_chunks']} chunks.")
        if not args.reindex:
            print("Use --reindex to force reprocessing.")
    else:
        if args.reindex:
            print("\nReindexing complete!")
        else:
            print("\nInitial processing complete!")
        print(f"Indexed {result['total_chunks']} chunks.")

    # Get embedding model and reranker from API
    embedding_model = api.embedding_model
    reranker = api.reranker
    database = api.database

    # Start query session
    query_session(
        database,
        embedding_model,
        reranker=reranker,
        n_results=args.results,
        reranker_multiplier=reranker_multiplier,
    )

    # Cleanup
    database.close()


if __name__ == "__main__":
    main()
