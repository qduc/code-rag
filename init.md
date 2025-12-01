This project converts a codebase into vector embeddings for easy searching and analysis.

It is a CLI tool that will be invoked from the codebase root directory.

At first run, it will check if the codebase has already been processed. If not, it will start processing the codebase.

## Technical Stack

- **Language**: Python
- **Vector Database**: Configurable with abstract interface
  - Default: ChromaDB (lightweight, embedded, runs in-process)
  - Other databases can be added later
  - Design: Use interface/abstract base class pattern for easy database swapping
- **Embedding Model**: Configurable with abstract interface
  - Default: Local model using sentence-transformers (e.g., all-MiniLM-L6-v2)
  - Extensible: Support for OpenAI embeddings and other providers
  - Design: Use interface/abstract base class pattern for easy model swapping

## Processing Steps

1. **File Discovery**: Recursively traverse the codebase to find all relevant source code files, ignoring specified directories (e.g., `node_modules`, `.git`, files ignored by `.gitignore`).

2. **File Reading**: Read the contents of each discovered file, handling different file encodings and large files appropriately. Convert the file into vector embeddings using the configured embedding model and store them in ChromaDB.

*This first version will use a naive file based chunking approach. Advanced chunking strategies (e.g., based on syntax, functions, classes) will be added in next version.*

## Usage

After the initial processing, the tool will start a query session where users can input a query and the tool will return relevant code snippets based on the vector embeddings.

The process of query and answer will be repeated until the user decides to exit the session (Ctrl+C).

## Architecture

- **Embedding Interface**: Abstract base class defining the embedding contract
- **Model Implementations**: Concrete implementations for different providers (local, OpenAI, etc.)
- **Configuration**: Support for model selection and configuration via config file or environment variables

## Notes

This is the first draft of the project. Further improvements and features will be added in future iterations.