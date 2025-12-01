"""Main CLI entry point for code-rag tool."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .config.config import Config
from .database.chroma_database import ChromaDatabase
from .database.qdrant_database import QdrantDatabase
from .database.database_interface import DatabaseInterface
from .embeddings.sentence_transformer_embedding import SentenceTransformerEmbedding
from .processor.file_processor import FileProcessor


def process_codebase(
    root_path: str,
    database: DatabaseInterface,
    embedding_model: SentenceTransformerEmbedding,
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
    embedding_model: SentenceTransformerEmbedding,
    n_results: int = 5,
) -> None:
    """
    Start an interactive query session.

    Args:
        database: Database instance to query
        embedding_model: Embedding model to generate query vectors
        n_results: Number of results to return per query
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

            # Search database
            results = database.query(query_embedding, n_results=n_results)

            # Display results
            if not results["documents"] or not results["documents"][0]:
                print("\nNo results found.")
                continue

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

                # Calculate similarity score (1 - distance for cosine)
                similarity = 1 - distance

                print("-" * 60)
                print(f"Result {i + 1} | Similarity: {similarity:.4f}")
                print(f"File: {file_path}")
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
        help="Type of vector database to use (default: qdrant)",
        default="qdrant",
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
    print("=" * 60)

    # Initialize database
    db_path = str(codebase_path / config.get_database_path())

    if args.database == "chroma":
        database = ChromaDatabase(persist_directory=db_path)
        database.initialize("codebase")
    elif args.database == "qdrant":
        database = QdrantDatabase(persist_directory=db_path)
        # Get embedding dimension from model
        temp_embedding = SentenceTransformerEmbedding(config.embedding_model)
        vector_size = temp_embedding.get_embedding_dimension()
        database.initialize("codebase", vector_size=vector_size)
    else:
        print(f"Error: Unsupported database type '{args.database}'")
        sys.exit(1)

    # Initialize embedding model (reuse if already loaded for Qdrant)
    print("\nLoading embedding model...")
    if args.database == "qdrant":
        embedding_model = temp_embedding
    else:
        embedding_model = SentenceTransformerEmbedding(config.embedding_model)
    print(f"Model loaded. Embedding dimension: {embedding_model.get_embedding_dimension()}")

    # Check if codebase needs processing
    if args.reindex or not database.is_processed():
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
    query_session(database, embedding_model, n_results=args.results)

    # Cleanup
    database.close()


if __name__ == "__main__":
    main()
