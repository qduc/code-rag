This project converts a codebase into vector embeddings for easy searching and analysis.

It is a CLI tool that will be invoked from the codebase root directory.

At first run, it will check if the codebase has already been processed. If not, it will start processing the codebase.

## Processing Steps

1. **File Discovery**: Recursively traverse the codebase to find all relevant source code files, ignoring specified directories (e.g., `node_modules`, `.git`, files ignored by `.gitignore`).

2. **File Reading**: Read the contents of each discovered file, handling different file encodings and large files appropriately. Convert the file into vector embeddings and store them in a vector database.

## Usage

After the initial processing, the tool will start a query session where users can input a query and the tool will return relevant code snippets based on the vector embeddings.

The process of query and answer will be repeated until the user decides to exit the session (Ctrl+C).

## Notes

This is the first draft of the project. Further improvements and features will be added in future iterations.