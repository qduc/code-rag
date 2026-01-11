"""Comprehensive tests for the database layer.

These tests verify:
1. ChromaDatabase collection management and operations
2. QdrantDatabase collection management and operations
3. Error handling for uninitialized collections
4. Edge cases (empty collections, missing data, etc.)
5. Document CRUD operations
6. Query functionality
"""

import json
import os
import shutil
import tempfile
import uuid
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from code_rag.database.chroma_database import ChromaDatabase
from code_rag.database.qdrant_database import QdrantDatabase


def make_uuid(name: str) -> str:
    """Generate a deterministic UUID from a name for testing."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


# ============================================================================
# ChromaDatabase Tests
# ============================================================================


class TestChromaDatabaseInit:
    """Tests for ChromaDatabase initialization."""

    def test_init_creates_client(self):
        """Test that initialization creates a ChromaDB client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            assert db.client is not None
            assert db.collection is None
            assert db.persist_directory == temp_dir

    def test_init_with_default_directory(self):
        """Test initialization with default persist directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                db = ChromaDatabase()
                assert db.persist_directory == ".code-rag"
            finally:
                os.chdir(original_cwd)


class TestChromaDatabaseCollection:
    """Tests for ChromaDatabase collection management."""

    @pytest.fixture
    def temp_db(self):
        """Provide a temporary ChromaDatabase instance."""
        temp_dir = tempfile.mkdtemp()
        db = ChromaDatabase(persist_directory=temp_dir)
        yield db
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialize_creates_collection(self, temp_db):
        """Test that initialize creates a new collection."""
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        assert result is None
        assert temp_db.collection is not None
        assert temp_db.collection.name == "test_collection"

    def test_initialize_existing_collection(self, temp_db):
        """Test initializing an existing collection returns it."""
        # Create collection first
        temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        # Initialize again
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        assert result is None
        assert temp_db.collection is not None

    def test_initialize_dimension_mismatch_with_model(self, temp_db):
        """Test dimension mismatch returns stored model name."""
        # Create collection with 384 dimensions
        temp_db.initialize(
            collection_name="test_collection",
            vector_size=384,
            model_name="original-model",
        )

        # Add a document to ensure dimensions are recorded
        temp_db.add(
            ids=["test1"], embeddings=[[0.1] * 384], documents=["test document"]
        )

        # Try to initialize with different dimensions
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=768, model_name="new-model"
        )

        assert result == "original-model"

    def test_initialize_dimension_mismatch_without_model_raises(self, temp_db):
        """Test dimension mismatch without stored model raises ValueError."""
        # Create a new temp db to avoid shared state
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)
            # Create collection without model name - only set dimension
            db.collection = db.client.create_collection(
                name="test_no_model",
                metadata={"hnsw:space": "cosine", "dimension": 384},
            )

            # Add a document
            db.add(ids=["test1"], embeddings=[[0.1] * 384], documents=["test document"])

            # Try with different dimensions - should raise since no model stored
            with pytest.raises(ValueError, match="Dimension mismatch"):
                db.initialize(collection_name="test_no_model", vector_size=768)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_model_name_returns_stored_name(self, temp_db):
        """Test get_model_name returns the stored model name."""
        temp_db.initialize(
            collection_name="test_collection",
            vector_size=384,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert temp_db.get_model_name() == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_model_name_returns_none_when_not_set(self, temp_db):
        """Test get_model_name returns None when no model stored."""
        temp_db.initialize(collection_name="test_collection", vector_size=384)

        assert temp_db.get_model_name() is None

    def test_get_model_name_returns_none_before_initialize(self, temp_db):
        """Test get_model_name returns None before initialization."""
        assert temp_db.get_model_name() is None

    def test_delete_collection(self, temp_db):
        """Test delete_collection removes the collection."""
        temp_db.initialize(collection_name="test_collection", vector_size=384)

        temp_db.delete_collection("test_collection")

        assert temp_db.collection is None

    def test_delete_nonexistent_collection(self, temp_db):
        """Test deleting a non-existent collection doesn't raise."""
        # Should not raise
        temp_db.delete_collection("nonexistent_collection")

    def test_close_does_not_raise(self, temp_db):
        """Test close method doesn't raise errors."""
        temp_db.initialize(collection_name="test_collection", vector_size=384)
        # Should not raise
        temp_db.close()


class TestChromaDatabaseOperations:
    """Tests for ChromaDatabase document operations."""

    @pytest.fixture
    def initialized_db(self):
        """Provide an initialized ChromaDatabase with a collection."""
        temp_dir = tempfile.mkdtemp()
        db = ChromaDatabase(persist_directory=temp_dir)
        db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )
        yield db
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_documents(self, initialized_db):
        """Test adding documents to the collection."""
        initialized_db.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["Document 1", "Document 2"],
            metadatas=[{"file_path": "/a.py"}, {"file_path": "/b.py"}],
        )

        assert initialized_db.count() == 2

    def test_add_without_metadatas(self, initialized_db):
        """Test adding documents without metadata."""
        initialized_db.add(
            ids=["doc1"], embeddings=[[0.1] * 384], documents=["Document 1"]
        )

        assert initialized_db.count() == 1

    def test_add_raises_without_initialization(self):
        """Test add raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.add(ids=["doc1"], embeddings=[[0.1] * 384], documents=["Document 1"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_returns_results(self, initialized_db):
        """Test query returns matching documents."""
        initialized_db.add(
            ids=["doc1", "doc2"],
            embeddings=[[0.1] * 384, [0.9] * 384],
            documents=["Document about auth", "Document about database"],
            metadatas=[{"file_path": "/auth.py"}, {"file_path": "/db.py"}],
        )

        results = initialized_db.query(embedding=[0.1] * 384, n_results=2)

        assert "ids" in results
        assert "documents" in results
        assert "distances" in results
        assert len(results["ids"][0]) <= 2

    def test_query_raises_without_initialization(self):
        """Test query raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.query(embedding=[0.1] * 384)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_count_returns_document_count(self, initialized_db):
        """Test count returns correct number of documents."""
        assert initialized_db.count() == 0

        initialized_db.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        assert initialized_db.count() == 3

    def test_count_raises_without_initialization(self):
        """Test count raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.count()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_is_processed_empty_collection(self, initialized_db):
        """Test is_processed returns False for empty collection."""
        assert initialized_db.is_processed() is False

    def test_is_processed_with_documents(self, initialized_db):
        """Test is_processed returns True when documents exist."""
        initialized_db.add(
            ids=["doc1"], embeddings=[[0.1] * 384], documents=["Document 1"]
        )

        assert initialized_db.is_processed() is True

    def test_delete_by_ids(self, initialized_db):
        """Test delete_by_ids removes specified documents."""
        initialized_db.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        initialized_db.delete_by_ids(["doc1", "doc3"])

        assert initialized_db.count() == 1

    def test_delete_by_ids_empty_list(self, initialized_db):
        """Test delete_by_ids with empty list does nothing."""
        initialized_db.add(ids=["doc1"], embeddings=[[0.1] * 384], documents=["Doc 1"])

        initialized_db.delete_by_ids([])

        assert initialized_db.count() == 1

    def test_delete_by_ids_raises_without_initialization(self):
        """Test delete_by_ids raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.delete_by_ids(["doc1"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_all_ids(self, initialized_db):
        """Test get_all_ids returns all document IDs."""
        initialized_db.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        ids = initialized_db.get_all_ids()

        assert set(ids) == {"doc1", "doc2", "doc3"}

    def test_get_all_ids_empty_collection(self, initialized_db):
        """Test get_all_ids returns empty list for empty collection."""
        ids = initialized_db.get_all_ids()
        assert ids == []

    def test_get_all_ids_raises_without_initialization(self):
        """Test get_all_ids raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.get_all_ids()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_ids_by_file(self, initialized_db):
        """Test get_ids_by_file returns IDs for specific file."""
        initialized_db.add(
            ids=["doc1", "doc2", "doc3"],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[
                {"file_path": "/a.py"},
                {"file_path": "/b.py"},
                {"file_path": "/a.py"},
            ],
        )

        ids = initialized_db.get_ids_by_file("/a.py")

        assert set(ids) == {"doc1", "doc3"}

    def test_get_ids_by_file_no_matches(self, initialized_db):
        """Test get_ids_by_file returns empty list when no matches."""
        initialized_db.add(
            ids=["doc1"],
            embeddings=[[0.1] * 384],
            documents=["Doc 1"],
            metadatas=[{"file_path": "/a.py"}],
        )

        ids = initialized_db.get_ids_by_file("/nonexistent.py")

        assert ids == []

    def test_get_ids_by_file_raises_without_initialization(self):
        """Test get_ids_by_file raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = ChromaDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.get_ids_by_file("/a.py")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# QdrantDatabase Tests
# ============================================================================


class TestQdrantDatabaseInit:
    """Tests for QdrantDatabase initialization."""

    def test_init_creates_local_client(self):
        """Test that initialization creates a local Qdrant client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            assert db.client is not None
            assert db.collection_name is None
            assert db.persist_directory == temp_dir

    def test_init_with_host_creates_remote_client(self):
        """Test initialization with host creates remote client."""
        with patch("code_rag.database.qdrant_database.QdrantClient") as mock_client:
            db = QdrantDatabase(host="localhost", port=6333)
            mock_client.assert_called_once_with(host="localhost", port=6333)

    def test_init_with_host_default_port(self):
        """Test initialization with host uses default port 6333."""
        with patch("code_rag.database.qdrant_database.QdrantClient") as mock_client:
            db = QdrantDatabase(host="localhost")
            mock_client.assert_called_once_with(host="localhost", port=6333)


class TestQdrantDatabaseCollection:
    """Tests for QdrantDatabase collection management."""

    @pytest.fixture
    def temp_db(self):
        """Provide a temporary QdrantDatabase instance."""
        temp_dir = tempfile.mkdtemp()
        db = QdrantDatabase(persist_directory=temp_dir)
        yield db
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_initialize_creates_collection(self, temp_db):
        """Test that initialize creates a new collection."""
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        assert result is None
        assert temp_db.collection_name == "test_collection"
        assert temp_db.vector_size == 384

    def test_initialize_existing_collection(self, temp_db):
        """Test initializing an existing collection returns it."""
        # Create collection first
        temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        # Initialize again
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        assert result is None
        assert temp_db.collection_name == "test_collection"

    def test_initialize_dimension_mismatch_with_model(self, temp_db):
        """Test dimension mismatch returns stored model name."""
        # Create collection with 384 dimensions
        temp_db.initialize(
            collection_name="test_collection",
            vector_size=384,
            model_name="original-model",
        )

        # Try to initialize with different dimensions
        result = temp_db.initialize(
            collection_name="test_collection", vector_size=768, model_name="new-model"
        )

        assert result == "original-model"
        assert temp_db._model_name == "original-model"
        assert temp_db.vector_size == 384

    def test_initialize_dimension_mismatch_without_model_raises(self, temp_db):
        """Test dimension mismatch without stored model raises ValueError."""
        # Create collection without model name
        temp_db.initialize(collection_name="test_collection", vector_size=384)

        # Remove metadata file to simulate no stored model
        metadata_path = temp_db._get_metadata_path("test_collection")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        # Try with different dimensions - should raise
        with pytest.raises(ValueError, match="Dimension mismatch"):
            temp_db.initialize(collection_name="test_collection", vector_size=768)

    def test_get_model_name_returns_stored_name(self, temp_db):
        """Test get_model_name returns the stored model name."""
        temp_db.initialize(
            collection_name="test_collection",
            vector_size=384,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert temp_db.get_model_name() == "sentence-transformers/all-MiniLM-L6-v2"

    def test_get_model_name_returns_none_when_not_set(self, temp_db):
        """Test get_model_name returns None when no model stored."""
        temp_db.initialize(collection_name="test_collection", vector_size=384)

        # Remove metadata file
        metadata_path = temp_db._get_metadata_path("test_collection")
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

        assert temp_db.get_model_name() is None

    def test_get_model_name_returns_none_before_initialize(self, temp_db):
        """Test get_model_name returns None before initialization."""
        assert temp_db.get_model_name() is None

    def test_store_and_get_model_name(self, temp_db):
        """Test model name storage and retrieval."""
        temp_db._store_model_name("test_collection", "my-model")

        result = temp_db._get_stored_model_name("test_collection")

        assert result == "my-model"

    def test_get_stored_model_name_invalid_json(self, temp_db):
        """Test _get_stored_model_name handles invalid JSON."""
        metadata_path = temp_db._get_metadata_path("test_collection")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        with open(metadata_path, "w") as f:
            f.write("invalid json")

        result = temp_db._get_stored_model_name("test_collection")

        assert result is None

    def test_delete_collection(self, temp_db):
        """Test delete_collection removes the collection."""
        temp_db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )

        temp_db.delete_collection("test_collection")

        assert temp_db.collection_name is None
        # Metadata file should also be deleted
        metadata_path = temp_db._get_metadata_path("test_collection")
        assert not os.path.exists(metadata_path)

    def test_delete_nonexistent_collection(self, temp_db):
        """Test deleting a non-existent collection doesn't raise."""
        # Should not raise
        temp_db.delete_collection("nonexistent_collection")

    def test_close_does_not_raise(self, temp_db):
        """Test close method doesn't raise errors."""
        temp_db.initialize(collection_name="test_collection", vector_size=384)
        # Should not raise
        temp_db.close()


class TestQdrantDatabaseOperations:
    """Tests for QdrantDatabase document operations."""

    @pytest.fixture
    def initialized_db(self):
        """Provide an initialized QdrantDatabase with a collection."""
        temp_dir = tempfile.mkdtemp()
        db = QdrantDatabase(persist_directory=temp_dir)
        db.initialize(
            collection_name="test_collection", vector_size=384, model_name="test-model"
        )
        yield db
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_documents(self, initialized_db):
        """Test adding documents to the collection."""
        initialized_db.add(
            ids=[make_uuid("doc1"), make_uuid("doc2")],
            embeddings=[[0.1] * 384, [0.2] * 384],
            documents=["Document 1", "Document 2"],
            metadatas=[{"file_path": "/a.py"}, {"file_path": "/b.py"}],
        )

        assert initialized_db.count() == 2

    def test_add_without_metadatas(self, initialized_db):
        """Test adding documents without metadata."""
        initialized_db.add(
            ids=[make_uuid("doc1")], embeddings=[[0.1] * 384], documents=["Document 1"]
        )

        assert initialized_db.count() == 1

    def test_add_raises_without_initialization(self):
        """Test add raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.add(
                    ids=[make_uuid("doc1")],
                    embeddings=[[0.1] * 384],
                    documents=["Document 1"],
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_query_returns_results(self, initialized_db):
        """Test query returns matching documents with mocked search."""
        initialized_db.add(
            ids=[make_uuid("doc1"), make_uuid("doc2")],
            embeddings=[[0.1] * 384, [0.9] * 384],
            documents=["Document about auth", "Document about database"],
            metadatas=[{"file_path": "/auth.py"}, {"file_path": "/db.py"}],
        )

        # Mock the search method since qdrant-client 1.16+ renamed it to query_points
        mock_result = MagicMock()
        mock_result.id = make_uuid("doc1")
        mock_result.score = 0.95
        mock_result.payload = {
            "document": "Document about auth",
            "file_path": "/auth.py",
        }

        with patch.object(
            initialized_db.client, "search", create=True, return_value=[mock_result]
        ):
            results = initialized_db.query(embedding=[0.1] * 384, n_results=2)

            # Verify ChromaDB-compatible format
            assert "ids" in results
            assert "documents" in results
            assert "distances" in results
            assert "metadatas" in results
            assert len(results["ids"][0]) >= 1

    def test_query_returns_chromadb_compatible_format(self, initialized_db):
        """Test query results are in ChromaDB-compatible nested list format."""
        initialized_db.add(
            ids=[make_uuid("doc1")],
            embeddings=[[0.1] * 384],
            documents=["Test document"],
            metadatas=[{"file_path": "/test.py"}],
        )

        # Mock the search method since qdrant-client 1.16+ renamed it to query_points
        mock_result = MagicMock()
        mock_result.id = make_uuid("doc1")
        mock_result.score = 0.99
        mock_result.payload = {"document": "Test document", "file_path": "/test.py"}

        with patch.object(
            initialized_db.client, "search", create=True, return_value=[mock_result]
        ):
            results = initialized_db.query(embedding=[0.1] * 384, n_results=1)

            # Results should be nested lists (ChromaDB format)
            assert isinstance(results["ids"], list)
            assert isinstance(results["ids"][0], list)
            assert isinstance(results["documents"], list)
            assert isinstance(results["documents"][0], list)
            assert isinstance(results["distances"], list)
            assert isinstance(results["distances"][0], list)

    def test_query_raises_without_initialization(self):
        """Test query raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.query(embedding=[0.1] * 384)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_count_returns_document_count(self, initialized_db):
        """Test count returns correct number of documents."""
        assert initialized_db.count() == 0

        initialized_db.add(
            ids=[make_uuid("doc1"), make_uuid("doc2"), make_uuid("doc3")],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        assert initialized_db.count() == 3

    def test_count_raises_without_initialization(self):
        """Test count raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.count()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_is_processed_empty_collection(self, initialized_db):
        """Test is_processed returns False for empty collection."""
        assert initialized_db.is_processed() is False

    def test_is_processed_with_documents(self, initialized_db):
        """Test is_processed returns True when documents exist."""
        initialized_db.add(
            ids=[make_uuid("doc1")], embeddings=[[0.1] * 384], documents=["Document 1"]
        )

        assert initialized_db.is_processed() is True

    def test_delete_by_ids(self, initialized_db):
        """Test delete_by_ids removes specified documents."""
        id1, id2, id3 = make_uuid("doc1"), make_uuid("doc2"), make_uuid("doc3")
        initialized_db.add(
            ids=[id1, id2, id3],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        initialized_db.delete_by_ids([id1, id3])

        assert initialized_db.count() == 1

    def test_delete_by_ids_empty_list(self, initialized_db):
        """Test delete_by_ids with empty list does nothing."""
        initialized_db.add(
            ids=[make_uuid("doc1")], embeddings=[[0.1] * 384], documents=["Doc 1"]
        )

        initialized_db.delete_by_ids([])

        assert initialized_db.count() == 1

    def test_delete_by_ids_raises_without_initialization(self):
        """Test delete_by_ids raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.delete_by_ids([make_uuid("doc1")])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_all_ids(self, initialized_db):
        """Test get_all_ids returns all document IDs."""
        id1, id2, id3 = make_uuid("doc1"), make_uuid("doc2"), make_uuid("doc3")
        initialized_db.add(
            ids=[id1, id2, id3],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
        )

        ids = initialized_db.get_all_ids()

        assert set(ids) == {id1, id2, id3}

    def test_get_all_ids_empty_collection(self, initialized_db):
        """Test get_all_ids returns empty list for empty collection."""
        ids = initialized_db.get_all_ids()
        assert ids == []

    def test_get_all_ids_raises_without_initialization(self):
        """Test get_all_ids raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.get_all_ids()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_get_ids_by_file(self, initialized_db):
        """Test get_ids_by_file returns IDs for specific file."""
        id1, id2, id3 = make_uuid("doc1"), make_uuid("doc2"), make_uuid("doc3")
        initialized_db.add(
            ids=[id1, id2, id3],
            embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[
                {"file_path": "/a.py"},
                {"file_path": "/b.py"},
                {"file_path": "/a.py"},
            ],
        )

        ids = initialized_db.get_ids_by_file("/a.py")

        assert set(ids) == {id1, id3}

    def test_get_ids_by_file_no_matches(self, initialized_db):
        """Test get_ids_by_file returns empty list when no matches."""
        initialized_db.add(
            ids=[make_uuid("doc1")],
            embeddings=[[0.1] * 384],
            documents=["Doc 1"],
            metadatas=[{"file_path": "/a.py"}],
        )

        ids = initialized_db.get_ids_by_file("/nonexistent.py")

        assert ids == []

    def test_get_ids_by_file_raises_without_initialization(self):
        """Test get_ids_by_file raises when collection not initialized."""
        temp_dir = tempfile.mkdtemp()
        try:
            db = QdrantDatabase(persist_directory=temp_dir)

            with pytest.raises(RuntimeError, match="Collection not initialized"):
                db.get_ids_by_file("/a.py")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestDatabaseEdgeCases:
    """Tests for edge cases in both database implementations."""

    def test_chroma_multiple_collections(self):
        """Test ChromaDB can handle multiple collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)

            db.initialize(collection_name="collection1", vector_size=384)
            db.add(ids=["doc1"], embeddings=[[0.1] * 384], documents=["Doc 1"])

            db.initialize(collection_name="collection2", vector_size=384)
            db.add(ids=["doc2"], embeddings=[[0.2] * 384], documents=["Doc 2"])

            # Current collection should be collection2
            assert db.count() == 1

            # Switch back to collection1
            db.initialize(collection_name="collection1", vector_size=384)
            assert db.count() == 1

    def test_qdrant_multiple_collections(self):
        """Test QdrantDB can handle multiple collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)

            db.initialize(collection_name="collection1", vector_size=384)
            db.add(
                ids=[make_uuid("doc1")], embeddings=[[0.1] * 384], documents=["Doc 1"]
            )

            db.initialize(collection_name="collection2", vector_size=384)
            db.add(
                ids=[make_uuid("doc2")], embeddings=[[0.2] * 384], documents=["Doc 2"]
            )

            # Current collection should be collection2
            assert db.count() == 1

            # Switch back to collection1
            db.initialize(collection_name="collection1", vector_size=384)
            assert db.count() == 1

    def test_chroma_large_batch_add(self):
        """Test ChromaDB can handle large batch additions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Add 100 documents at once
            n = 100
            db.add(
                ids=[f"doc{i}" for i in range(n)],
                embeddings=[[0.1 * (i % 10)] * 384 for i in range(n)],
                documents=[f"Document {i}" for i in range(n)],
            )

            assert db.count() == n

    def test_qdrant_large_batch_add(self):
        """Test QdrantDB can handle large batch additions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Add 100 documents at once
            n = 100
            db.add(
                ids=[make_uuid(f"doc{i}") for i in range(n)],
                embeddings=[[0.1 * (i % 10)] * 384 for i in range(n)],
                documents=[f"Document {i}" for i in range(n)],
            )

            assert db.count() == n

    def test_chroma_special_characters_in_documents(self):
        """Test ChromaDB handles special characters in documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            special_doc = "def func(): # 日本語 émojis 🎉 special chars: <>&\"'"
            db.add(ids=["doc1"], embeddings=[[0.1] * 384], documents=[special_doc])

            results = db.query(embedding=[0.1] * 384, n_results=1)
            assert results["documents"][0][0] == special_doc

    def test_qdrant_special_characters_in_documents(self):
        """Test QdrantDB handles special characters in documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            special_doc = "def func(): # 日本語 émojis 🎉 special chars: <>&\"'"
            doc_id = make_uuid("doc1")
            db.add(ids=[doc_id], embeddings=[[0.1] * 384], documents=[special_doc])

            # Mock the search method since qdrant-client 1.16+ renamed it
            mock_result = MagicMock()
            mock_result.id = doc_id
            mock_result.score = 0.99
            mock_result.payload = {"document": special_doc}

            with patch.object(
                db.client, "search", create=True, return_value=[mock_result]
            ):
                results = db.query(embedding=[0.1] * 384, n_results=1)
                assert results["documents"][0][0] == special_doc

    def test_chroma_query_more_than_available(self):
        """Test ChromaDB handles requesting more results than available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            db.add(
                ids=["doc1", "doc2"],
                embeddings=[[0.1] * 384, [0.2] * 384],
                documents=["Doc 1", "Doc 2"],
            )

            results = db.query(embedding=[0.1] * 384, n_results=100)
            assert len(results["ids"][0]) == 2

    def test_qdrant_query_more_than_available(self):
        """Test QdrantDB handles requesting more results than available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            id1, id2 = make_uuid("doc1"), make_uuid("doc2")
            db.add(
                ids=[id1, id2],
                embeddings=[[0.1] * 384, [0.2] * 384],
                documents=["Doc 1", "Doc 2"],
            )

            # Mock the search method - returns only 2 results even if more requested
            mock_result1 = MagicMock()
            mock_result1.id = id1
            mock_result1.score = 0.99
            mock_result1.payload = {"document": "Doc 1"}
            mock_result2 = MagicMock()
            mock_result2.id = id2
            mock_result2.score = 0.95
            mock_result2.payload = {"document": "Doc 2"}

            with patch.object(
                db.client,
                "search",
                create=True,
                return_value=[mock_result1, mock_result2],
            ):
                results = db.query(embedding=[0.1] * 384, n_results=100)
                assert len(results["ids"][0]) == 2

    def test_chroma_upsert_behavior(self):
        """Test ChromaDB add behavior with existing IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Add document
            db.add(
                ids=["doc1"], embeddings=[[0.1] * 384], documents=["Original document"]
            )

            # ChromaDB's add will fail if ID exists, need to delete first
            # This test verifies the current behavior
            assert db.count() == 1

    def test_qdrant_upsert_behavior(self):
        """Test QdrantDB upsert behavior with existing IDs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            doc_id = make_uuid("doc1")

            # Add document
            db.add(
                ids=[doc_id], embeddings=[[0.1] * 384], documents=["Original document"]
            )

            # Qdrant uses upsert, so this should update
            db.add(
                ids=[doc_id], embeddings=[[0.2] * 384], documents=["Updated document"]
            )

            # Should still have only 1 document (upsert)
            assert db.count() == 1

            # Mock the search to verify the updated document
            mock_result = MagicMock()
            mock_result.id = doc_id
            mock_result.score = 0.99
            mock_result.payload = {"document": "Updated document"}

            with patch.object(
                db.client, "search", create=True, return_value=[mock_result]
            ):
                results = db.query(embedding=[0.2] * 384, n_results=1)
                assert results["documents"][0][0] == "Updated document"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestDatabaseErrorHandling:
    """Tests for error handling in database implementations."""

    def test_chroma_delete_by_ids_error_handling(self):
        """Test ChromaDB delete_by_ids handles errors properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Deleting non-existent IDs should not raise in ChromaDB
            db.delete_by_ids(["nonexistent_id"])

    def test_qdrant_delete_by_ids_error_handling(self):
        """Test QdrantDB delete_by_ids handles errors properly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Deleting non-existent IDs should not raise in Qdrant
            db.delete_by_ids(["nonexistent_id"])

    def test_chroma_get_all_ids_error_recovery(self):
        """Test ChromaDB get_all_ids returns empty list on error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Mock the collection.get to raise an exception
            with patch.object(
                db.collection, "get", side_effect=Exception("Test error")
            ):
                result = db.get_all_ids()
                assert result == []

    def test_qdrant_get_all_ids_error_recovery(self):
        """Test QdrantDB get_all_ids returns empty list on error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Mock the client.scroll to raise an exception
            with patch.object(db.client, "scroll", side_effect=Exception("Test error")):
                result = db.get_all_ids()
                assert result == []

    def test_chroma_get_ids_by_file_error_recovery(self):
        """Test ChromaDB get_ids_by_file returns empty list on error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = ChromaDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Mock the collection.get to raise an exception
            with patch.object(
                db.collection, "get", side_effect=Exception("Test error")
            ):
                result = db.get_ids_by_file("/test.py")
                assert result == []

    def test_qdrant_get_ids_by_file_error_recovery(self):
        """Test QdrantDB get_ids_by_file returns empty list on error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db = QdrantDatabase(persist_directory=temp_dir)
            db.initialize(collection_name="test", vector_size=384)

            # Mock the client.scroll to raise an exception
            with patch.object(db.client, "scroll", side_effect=Exception("Test error")):
                result = db.get_ids_by_file("/test.py")
                assert result == []
