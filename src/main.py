"""Main CLI entry point for code-rag tool."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config.config import Config
from .database.chroma_database import ChromaDatabase
from .database.qdrant_database import QdrantDatabase
from .database.database_interface import DatabaseInterface
from .embeddings.embedding_interface import EmbeddingInterface
from .embeddings.openai_embedding import OpenAIEmbedding
from .embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from .processor.file_processor import FileProcessor
from .reranker.reranker_interface import RerankerInterface
from .reranker.cross_encoder_reranker import CrossEncoderReranker
from pathlib import PurePath


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
    processor = FileProcessor()

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

            # Generate embedding for query
            query_embedding = embedding_model.embed(query)

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
                    reranked_indices = reranker.rerank(query, documents, top_k=n_results)

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

            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                file_path = metadata.get("file_path", "Unknown")
                chunk_index = metadata.get("chunk_index", 0)
                total_chunks = metadata.get("total_chunks", 1)
                start_line = metadata.get("start_line")
                end_line = metadata.get("end_line")

                # Calculate similarity score
                if reranker is not None:
                    similarity = distance  # Already a relevance score (higher = better)
                else:
                    similarity = 1 - distance  # Convert cosine distance to similarity

                print("-" * 60)
                print(f"Result {i + 1} | Similarity: {similarity:.4f}")
                print(f"File: {file_path}")
                if start_line and end_line:
                    print(f"Lines: {start_line}-{end_line} | Chunk: {chunk_index + 1}/{total_chunks}")
                else:
                    print(f"Chunk: {chunk_index + 1}/{total_chunks}")
                print("-" * 60)

                # Truncate long documents for display
                display_doc = doc[:500] + "..." if len(doc) > 500 else doc
                print(display_doc)
                print()

        except KeyboardInterrupt:
            print("\n\nExiting query session. Goodbye!")
            break
        except Exception as e:
            print(f"\nError during query: {e}")

def looks_like_codebase(root_path: Path, processor: FileProcessor) -> bool:
    """Heuristic: return True if the folder likely contains a codebase.

    Checks for common repo files/dirs, then falls back to counting discovered source files.
    """
    markers = [
        "setup.py",
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "Cargo.toml",
        "go.mod",
        ".git",
        "src",
    ]

    try:
        for m in markers:
            if (root_path / m).exists():
                return True

        # Fallback: if discover_files finds a reasonable number of source files,
        # consider it a codebase. Small threshold avoids misdetecting home dirs.
        files = processor.discover_files(str(root_path))
        return len(files) >= 5
    except Exception:
        # On any unexpected error, be conservative and treat as non-codebase
        return False

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

    # Initialize database
    db_path = config.get_database_path()

    def create_embedding_model(model_name: str) -> EmbeddingInterface:
        """Create an embedding model instance based on the model name."""
        if model_name.startswith("text-embedding-"):
            return OpenAIEmbedding(model_name)
        else:
            return SentenceTransformerEmbedding(model_name)

    # Initialize embedding model
    print("\nLoading embedding model...")
    embedding_model = create_embedding_model(config.embedding_model)
    print(f"Model loaded. Embedding dimension: {embedding_model.get_embedding_dimension()}")

    # Initialize reranker
    reranker = None
    reranker_enabled = config.is_reranker_enabled() and not args.no_rerank

    if reranker_enabled:
        print("\nLoading reranker model...")
        reranker_model_name = args.reranker_model or config.get_reranker_model()
        try:
            reranker = CrossEncoderReranker(reranker_model_name)
            print(f"Reranker loaded: {reranker_model_name}")
        except Exception as e:
            print(f"Warning: Failed to load reranker ({e}), disabling reranking")
            reranker = None
    else:
        print("\nReranking disabled")

    # Get reranker multiplier
    reranker_multiplier = args.reranker_multiplier or config.get_reranker_multiplier()

    if args.database == "chroma":
        database = ChromaDatabase(persist_directory=db_path)
        vector_size = embedding_model.get_embedding_dimension()
        if args.reindex:
            database.delete_collection("codebase")
        stored_model = database.initialize(
            "codebase", vector_size=vector_size, model_name=config.embedding_model
        )
        # If dimension mismatch, reload with the stored model
        if stored_model and stored_model != config.embedding_model:
            print(f"\nReloading with model: {stored_model}")
            embedding_model = create_embedding_model(stored_model)
            print(f"Model loaded. Embedding dimension: {embedding_model.get_embedding_dimension()}")
    elif args.database == "qdrant":
        database = QdrantDatabase(persist_directory=db_path)
        vector_size = embedding_model.get_embedding_dimension()
        if args.reindex:
            database.delete_collection("codebase")
        stored_model = database.initialize(
            "codebase", vector_size=vector_size, model_name=config.embedding_model
        )
        # If dimension mismatch, reload with the stored model
        if stored_model and stored_model != config.embedding_model:
            print(f"\nReloading with model: {stored_model}")
            embedding_model = create_embedding_model(stored_model)
            print(f"Model loaded. Embedding dimension: {embedding_model.get_embedding_dimension()}")
    else:
        print(f"Error: Unsupported database type '{args.database}'")
        sys.exit(1)

    # Check if codebase needs processing
    if args.reindex or not database.is_processed():
        processor = FileProcessor()

        # If the folder doesn't look like a codebase, ask for confirmation
        if not looks_like_codebase(codebase_path, processor):
            print("\nWarning: The target directory does not appear to contain a typical codebase.")
            try:
                resp = input("Continue indexing this directory? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                print("\nAborting.")
                sys.exit(1)

            if resp not in ("y", "yes"):
                print("Aborting indexing.")
                sys.exit(0)

        if args.reindex:
            print("\nReindexing codebase...")
        else:
            print("\nCodebase not yet processed. Starting initial processing...")

        total_chunks = process_codebase(
            str(codebase_path),
            database,
            embedding_model,
            config,
        )

        print(f"\nProcessing complete! Indexed {total_chunks} chunks.")
    else:
        print(f"\nCodebase already processed. Found {database.count()} chunks.")
        print("Use --reindex to force reprocessing.")

    # Start query session
    query_session(
        database,
        embedding_model,
        reranker=reranker,
        n_results=args.results,
        reranker_multiplier=reranker_multiplier
    )

    # Cleanup
    database.close()


if __name__ == "__main__":
    main()
