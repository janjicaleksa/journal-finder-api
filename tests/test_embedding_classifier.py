"""
Unit tests for EmbeddingClassifier.

SentenceTransformer is mocked throughout so tests run without downloading
any model weights. The mock's encode() returns deterministic unit-normalised
numpy vectors whose cosine similarity (dot product) is fully predictable.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.services.embedding_classifier import EmbeddingClassifier


JOURNAL_DESCRIPTIONS = {
    "biology": "Study of living organisms, cells, genetics, DNA and proteins.",
    "physics": "Study of matter, energy, quantum mechanics and particle physics.",
    "chemistry": "Study of chemical compounds, reactions, catalysts and polymers.",
}

# Pre-built orthogonal unit vectors – one per journal + one for abstracts.
# Similarity: abstract_vec is identical to biology_vec → biology wins.
BIOLOGY_VEC  = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
PHYSICS_VEC  = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
CHEMISTRY_VEC = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

JOURNAL_MATRIX = np.vstack([BIOLOGY_VEC, PHYSICS_VEC, CHEMISTRY_VEC])  # (3, 3)

# Abstract that should match biology perfectly (dot = 1.0 with biology, 0 elsewhere)
ABSTRACT_BIOLOGY_VEC = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
# Abstract that won't match anything
ABSTRACT_ZERO_VEC = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


def make_mock_model(abstract_vec: np.ndarray = ABSTRACT_BIOLOGY_VEC) -> MagicMock:
    """Return a mock SentenceTransformer whose encode() is controllable.

    Call order:
      1st call  – __init__ encodes all journal descriptions → JOURNAL_MATRIX
      2nd call  – classify() encodes the single abstract   → abstract_vec
      3rd+ call – get_top_supporting_sentences() encodes N sentences → N rows
    """
    model = MagicMock()
    call_count = [0]

    def encode_side_effect(inputs, normalize_embeddings=True):
        call_count[0] += 1
        if call_count[0] == 1:
            return JOURNAL_MATRIX
        n = len(inputs)
        if n == 1:
            return abstract_vec
        # Multiple sentences: repeat abstract_vec to produce (n, dim) matrix
        return np.tile(abstract_vec, (n, 1))

    model.encode.side_effect = encode_side_effect
    return model


@pytest.fixture
def clf(mock_model):
    """EmbeddingClassifier with the injected mock model."""
    return EmbeddingClassifier.__new__(EmbeddingClassifier)


@pytest.fixture
def mock_model():
    return make_mock_model()


# ---------------------------------------------------------------------------
# Helpers – patch SentenceTransformer for a whole test class
# ---------------------------------------------------------------------------


def build_classifier(abstract_vec=ABSTRACT_BIOLOGY_VEC):
    """Instantiate EmbeddingClassifier with a mocked SentenceTransformer."""
    with patch("app.services.embedding_classifier.SentenceTransformer") as MockST:
        MockST.return_value = make_mock_model(abstract_vec)
        return EmbeddingClassifier(JOURNAL_DESCRIPTIONS, embedding_model_name="mock-model")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestEmbeddingClassifierInit:
    def test_valid_descriptions_accepted(self):
        clf = build_classifier()
        assert clf.journal_labels == list(JOURNAL_DESCRIPTIONS.keys())

    def test_journal_embeddings_shape(self):
        clf = build_classifier()
        assert clf.journal_embeddings.shape == (len(JOURNAL_DESCRIPTIONS), 3)

    def test_model_name_stored(self):
        clf = build_classifier()
        assert clf.embedding_model_name == "mock-model"

    def test_empty_dict_raises(self):
        with patch("app.services.embedding_classifier.SentenceTransformer"):
            with pytest.raises(ValueError, match="Journal descriptions must not be empty"):
                EmbeddingClassifier({}, embedding_model_name="mock-model")

    def test_none_raises(self):
        with patch("app.services.embedding_classifier.SentenceTransformer"):
            with pytest.raises((ValueError, TypeError)):
                EmbeddingClassifier(None, embedding_model_name="mock-model")


# ---------------------------------------------------------------------------
# classify – input validation
# ---------------------------------------------------------------------------


class TestEmbeddingClassifyInputValidation:
    def setup_method(self):
        self.clf = build_classifier()

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Abstract must not be empty"):
            self.clf.classify("")

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            self.clf.classify(None)


# ---------------------------------------------------------------------------
# classify – return structure
# ---------------------------------------------------------------------------


class TestEmbeddingClassifyReturnStructure:
    def setup_method(self):
        self.clf = build_classifier()

    def test_returns_dict(self):
        result = self.clf.classify("some abstract about cells")
        assert isinstance(result, dict)

    def test_has_predicted_category_key(self):
        result = self.clf.classify("some abstract")
        assert "predicted_category" in result

    def test_has_scores_key(self):
        result = self.clf.classify("some abstract")
        assert "scores" in result

    def test_scores_length_equals_number_of_journals(self):
        result = self.clf.classify("some abstract")
        assert len(result["scores"]) == len(JOURNAL_DESCRIPTIONS)

    def test_each_score_entry_has_required_keys(self):
        result = self.clf.classify("some abstract")
        for entry in result["scores"]:
            assert "label" in entry
            assert "score" in entry

    def test_scores_sorted_descending(self):
        result = self.clf.classify("some abstract")
        scores = [e["score"] for e in result["scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_score_is_float(self):
        result = self.clf.classify("some abstract")
        for entry in result["scores"]:
            assert isinstance(entry["score"], float)


# ---------------------------------------------------------------------------
# classify – scoring correctness
# ---------------------------------------------------------------------------


class TestEmbeddingClassifyScoring:
    def test_highest_similarity_journal_is_predicted(self):
        # abstract vec == biology vec → biology wins with score 1.0
        clf = build_classifier(ABSTRACT_BIOLOGY_VEC)
        result = clf.classify("abstract text doesn't matter – mock controls similarity")
        assert result["predicted_category"] == "biology"

    def test_perfect_similarity_score_is_1(self):
        clf = build_classifier(ABSTRACT_BIOLOGY_VEC)
        result = clf.classify("abstract text")
        biology_entry = next(e for e in result["scores"] if e["label"] == "biology")
        assert biology_entry["score"] == pytest.approx(1.0)

    def test_orthogonal_journals_score_0(self):
        clf = build_classifier(ABSTRACT_BIOLOGY_VEC)
        result = clf.classify("abstract text")
        for label in ("physics", "chemistry"):
            entry = next(e for e in result["scores"] if e["label"] == label)
            assert entry["score"] == pytest.approx(0.0)

    def test_zero_vector_returns_none_predicted_category(self):
        clf = build_classifier(ABSTRACT_ZERO_VEC)
        result = clf.classify("abstract text")
        assert result["predicted_category"] is None

    def test_score_rounded_to_2_decimals(self):
        # Use a vector with a non-trivial dot product
        partial_vec = np.array([[0.7, 0.5, 0.5]], dtype=np.float32)
        clf = build_classifier(partial_vec)
        result = clf.classify("abstract text")
        for entry in result["scores"]:
            assert entry["score"] == round(entry["score"], 2)

    def test_physics_wins_when_abstract_matches_physics(self):
        clf = build_classifier(PHYSICS_VEC)
        result = clf.classify("abstract text")
        assert result["predicted_category"] == "physics"

    def test_chemistry_wins_when_abstract_matches_chemistry(self):
        clf = build_classifier(CHEMISTRY_VEC)
        result = clf.classify("abstract text")
        assert result["predicted_category"] == "chemistry"


# ---------------------------------------------------------------------------
# classify – single-journal edge case
# ---------------------------------------------------------------------------


class TestEmbeddingClassifySingleJournal:
    def test_single_journal_match(self):
        descriptions = {"medicine": "diseases viruses vaccines"}
        with patch("app.services.embedding_classifier.SentenceTransformer") as MockST:
            model = MagicMock()
            vec = np.array([[1.0, 0.0]], dtype=np.float32)
            model.encode.return_value = vec
            MockST.return_value = model

            clf = EmbeddingClassifier(descriptions, embedding_model_name="mock")
            result = clf.classify("some abstract")
            assert result["predicted_category"] == "medicine"

    def test_single_journal_no_match(self):
        descriptions = {"medicine": "diseases viruses vaccines"}
        with patch("app.services.embedding_classifier.SentenceTransformer") as MockST:
            model = MagicMock()
            journal_vec = np.array([[1.0, 0.0]], dtype=np.float32)
            abstract_vec = np.array([[0.0, 0.0]], dtype=np.float32)

            def encode_side_effect(inputs, normalize_embeddings=True):
                if isinstance(inputs, list) and len(inputs) == 1 and inputs[0] == "diseases viruses vaccines":
                    return journal_vec
                return abstract_vec

            model.encode.side_effect = encode_side_effect
            MockST.return_value = model

            clf = EmbeddingClassifier(descriptions, embedding_model_name="mock")
            result = clf.classify("unrelated abstract")
            assert result["predicted_category"] is None


# ---------------------------------------------------------------------------
# get_top_supporting_sentences
# ---------------------------------------------------------------------------


class TestGetTopSupportingSentences:
    def _build_clf_with_sentence_encode(self, sentence_scores: list[float]):
        """
        Build a classifier whose encode() returns:
          - journal matrix on init
          - per-sentence vectors that produce the given dot-product scores
            against the biology journal vector (BIOLOGY_VEC = [1,0,0])
        """
        with patch("app.services.embedding_classifier.SentenceTransformer") as MockST:
            model = MagicMock()
            call_count = [0]

            def encode_side_effect(inputs, normalize_embeddings=True):
                call_count[0] += 1
                if call_count[0] == 1:
                    # Init call – return journal matrix
                    return JOURNAL_MATRIX
                # classify() call → single abstract embedding
                if isinstance(inputs, list) and len(inputs) == 1:
                    return ABSTRACT_BIOLOGY_VEC
                # get_top_supporting_sentences call → one vector per sentence
                n = len(inputs)
                # Build vectors so dot(vec_i, biology_vec) == sentence_scores[i]
                vecs = np.zeros((n, 3), dtype=np.float32)
                for i, s in enumerate(sentence_scores[:n]):
                    vecs[i, 0] = s  # dot with [1,0,0] == s
                return vecs

            model.encode.side_effect = encode_side_effect
            MockST.return_value = model
            clf = EmbeddingClassifier(JOURNAL_DESCRIPTIONS, embedding_model_name="mock")
            # Trigger the single-abstract encode so call_count increments correctly
            clf.model = model
            return clf

    def test_returns_list(self):
        clf = build_classifier()
        sentences = clf.get_top_supporting_sentences(
            "Cells are fundamental. DNA carries genetic information.", "biology"
        )
        assert isinstance(sentences, list)

    def test_each_entry_has_sentence_and_score_keys(self):
        clf = build_classifier()
        sentences = clf.get_top_supporting_sentences(
            "Cells are fundamental. DNA carries genetic information.", "biology"
        )
        for entry in sentences:
            assert "sentence" in entry
            assert "score" in entry

    def test_default_top_k_is_2(self):
        clf = build_classifier()
        abstract = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = clf.get_top_supporting_sentences(abstract, "biology")
        assert len(result) <= 2

    def test_custom_top_k_respected(self):
        clf = build_classifier()
        abstract = "Sentence one. Sentence two. Sentence three. Sentence four."
        result = clf.get_top_supporting_sentences(abstract, "biology", top_k=3)
        assert len(result) <= 3

    def test_empty_abstract_returns_empty_list(self):
        clf = build_classifier()
        result = clf.get_top_supporting_sentences("", "biology")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        clf = build_classifier()
        result = clf.get_top_supporting_sentences("   ", "biology")
        assert result == []

    def test_single_sentence_abstract(self):
        clf = build_classifier()
        result = clf.get_top_supporting_sentences("Only one sentence here.", "biology")
        assert len(result) <= 1

    def test_sentences_sorted_by_score_descending(self):
        clf = build_classifier()
        abstract = "Cells are fundamental. Quantum mechanics rules. DNA carries information."
        result = clf.get_top_supporting_sentences(abstract, "biology", top_k=3)
        if len(result) > 1:
            scores = [e["score"] for e in result]
            assert scores == sorted(scores, reverse=True)
