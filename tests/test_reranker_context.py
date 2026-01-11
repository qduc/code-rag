from unittest.mock import patch

from code_rag.reranker.cross_encoder_reranker import CrossEncoderReranker


def test_cross_encoder_reranker_with_context():
    # Mock the CrossEncoder model
    with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as MockEncoder:
        mock_instance = MockEncoder.return_value
        mock_instance.predict.return_value = [0.9]

        reranker = CrossEncoderReranker(lazy_load=False)

        query = "how to validate age"
        documents = ["if (x > 10) { return true; }"]
        metadatas = [
            {
                "file_path": "utils.js",
                "function_name": "validateAge",
                "class_name": "Validator",
            }
        ]

        reranker.rerank(query, documents, metadatas=metadatas)

        # Check what was passed to predict
        expected_text = "File: utils.js | Class: Validator | Function: validateAge | Code: if (x > 10) { return true; }"

        args, kwargs = mock_instance.predict.call_args
        pairs = args[0]

        assert pairs[0][0] == query
        assert pairs[0][1] == expected_text


def test_cross_encoder_reranker_partial_metadata():
    with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as MockEncoder:
        mock_instance = MockEncoder.return_value
        mock_instance.predict.return_value = [0.8]

        reranker = CrossEncoderReranker(lazy_load=False)

        query = "search"
        documents = ["print('hello')"]
        metadatas = [
            {
                "file_path": "main.py"
                # function_name and class_name missing
            }
        ]

        reranker.rerank(query, documents, metadatas=metadatas)

        expected_text = "File: main.py | Code: print('hello')"

        args, kwargs = mock_instance.predict.call_args
        pairs = args[0]

        assert pairs[0][1] == expected_text


def test_cross_encoder_reranker_no_metadata():
    with patch("code_rag.reranker.cross_encoder_reranker.CrossEncoder") as MockEncoder:
        mock_instance = MockEncoder.return_value
        mock_instance.predict.return_value = [0.7]

        reranker = CrossEncoderReranker(lazy_load=False)

        query = "search"
        documents = ["no context here"]

        reranker.rerank(query, documents)

        # Should fall back to just document
        args, kwargs = mock_instance.predict.call_args
        pairs = args[0]

        assert pairs[0][1] == "no context here"
