"""Setup script for code-rag."""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="code-rag",
    version="0.1.0",
    description="Convert a codebase into vector embeddings for searching and analysis",
    author="",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "chromadb>=1.3.5",
        "qdrant-client>=1.7.0",
        "sentence-transformers>=2.2.0",
        "einops>=0.6.0",
        "openai>=1.50.0",
        "python-dotenv>=0.21.0",
        "mcp>=0.9.0",
        "tree-sitter>=0.22.0",
        "tree-sitter-python>=0.21.0",
        "tree-sitter-javascript>=0.21.0",
        "tree-sitter-typescript>=0.21.0",
        "tree-sitter-go>=0.21.0",
        "tree-sitter-rust>=0.21.0",
        "tree-sitter-java>=0.21.0",
        "tree-sitter-cpp>=0.21.0",
        "tree-sitter-c>=0.21.0",
    ],
    entry_points={
        "console_scripts": [
            "code-rag=src.main:main",
            "code-rag-mcp=src.mcp_server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
