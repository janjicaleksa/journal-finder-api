"""
Session-wide mock for sentence_transformers.

routes.py constructs an EmbeddingClassifier at module level, which calls
SentenceTransformer() and loads model weights. By injecting a mock into
sys.modules here (conftest module-level code runs before any test-module
import), we prevent any real model download during the test suite.

Existing test_embedding_classifier.py tests are unaffected: they patch
the local 'SentenceTransformer' reference in app.services.embedding_classifier,
which correctly overrides whatever was imported during module load.
"""
import sys
import numpy as np
from unittest.mock import MagicMock

if "sentence_transformers" not in sys.modules:
    _mock_instance = MagicMock()
    # JOURNAL_DESCRIPTIONS has 4 entries → encode must return shape (4, N)
    _mock_instance.encode.return_value = np.zeros((4, 384), dtype=np.float32)

    _mock_class = MagicMock(return_value=_mock_instance)

    _mock_module = MagicMock()
    _mock_module.SentenceTransformer = _mock_class

    sys.modules["sentence_transformers"] = _mock_module
