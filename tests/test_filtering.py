import tempfile
from pathlib import Path

from code_rag.api import CodeRAGAPI


def create_dummy_codebase(base_path):
    """Creates a dummy codebase structure."""
    files = {
        "src/main.py": "def main():\n    print('Hello from main')\n",
        "src/utils.py": "def helper():\n    print('I am a helper')\n",
        "docs/readme.md": "# Readme\nThis is the main documentation.\n",
        "tests/test_main.py": "def test_main():\n    assert True\n",
    }

    for rel_path, content in files.items():
        file_path = base_path / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)


def test_filtering():
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        codebase_path = temp_path / "dummy_codebase"
        db_path = temp_path / "db"

        create_dummy_codebase(codebase_path)

        print(f"Created dummy codebase at {codebase_path}")

        # Initialize API
        # Disable reranker to speed up tests and avoid downloading another model
        api = CodeRAGAPI(database_path=str(db_path), reranker_enabled=False)

        # Index codebase
        print("Indexing codebase...")
        # We use ensure_indexed which handles collection creation
        result = api.ensure_indexed(str(codebase_path))
        if not result["success"]:
            print(f"Indexing failed: {result.get('error')}")
            return

        print("Indexing complete. Running tests...")

        # Test 1: Search "main" with file_types=[".py"]
        print("\nTest 1: Search 'main' with file_types=['.py']")
        results = api.search("main", file_types=[".py"])
        paths = [r["file_path"] for r in results]
        print(f"Results: {paths}")

        # Should contain src/main.py and tests/test_main.py
        # Should NOT contain docs/readme.md
        # Note: paths are absolute, so we check for suffix
        assert any(p.endswith("src/main.py") for p in paths), "src/main.py missing"
        assert any(
            p.endswith("tests/test_main.py") for p in paths
        ), "tests/test_main.py missing"
        assert not any(
            p.endswith("docs/readme.md") for p in paths
        ), "docs/readme.md should not be present"
        print("Test 1 PASSED")

        # Test 2: Search "main" with include_paths=["src/"]
        print("\nTest 2: Search 'main' with include_paths=['src/']")
        results = api.search("main", include_paths=["src/"])
        paths = [r["file_path"] for r in results]
        print(f"Results: {paths}")

        # Should contain src/main.py
        # Should NOT contain tests/test_main.py
        assert any(p.endswith("src/main.py") for p in paths), "src/main.py missing"
        assert not any(
            p.endswith("tests/test_main.py") for p in paths
        ), "tests/test_main.py should not be present"
        print("Test 2 PASSED")

        # Test 3: Search "main" with file_types=[".md"]
        print("\nTest 3: Search 'main' with file_types=['.md']")
        results = api.search("main", file_types=[".md"])
        paths = [r["file_path"] for r in results]
        print(f"Results: {paths}")

        # Should contain docs/readme.md
        assert any(
            p.endswith("docs/readme.md") for p in paths
        ), "docs/readme.md missing"
        print("Test 3 PASSED")

        api.close()


if __name__ == "__main__":
    test_filtering()
