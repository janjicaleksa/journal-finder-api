import pytest

from app.services.tfidf_classifier import TfidfClassifier

JOURNAL_DESCRIPTIONS = {
    "biology": "Study of living organisms, cells, genetics, DNA, proteins and ecosystems.",
    "physics": "Study of matter, energy, quantum mechanics, relativity and particle physics.",
    "chemistry": "Study of chemical compounds, molecular reactions, catalysts and polymers.",
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestTfidfClassifierInit:
    def test_valid_descriptions_accepted(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)
        assert clf.journal_labels == list(JOURNAL_DESCRIPTIONS.keys())

    def test_journal_vectors_are_computed(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)
        assert clf.journal_vectors is not None
        assert clf.journal_vectors.shape[0] == len(JOURNAL_DESCRIPTIONS)

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Journal descriptions must not be empty"):
            TfidfClassifier({})

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            TfidfClassifier(None)

    def test_custom_ngram_range_stored(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS, ngram_range=(1, 3))
        assert clf.ngram_range == (1, 3)

    def test_custom_max_reasoning_terms_stored(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS, max_reasoning_terms=10)
        assert clf.max_reasoning_terms == 10

    def test_default_max_reasoning_terms(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)
        assert clf.max_reasoning_terms == 5


# ---------------------------------------------------------------------------
# classify – input validation
# ---------------------------------------------------------------------------


class TestTfidfClassifyInputValidation:
    def setup_method(self):
        self.clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Abstract must not be empty"):
            self.clf.classify("")

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            self.clf.classify(None)


# ---------------------------------------------------------------------------
# classify – return structure
# ---------------------------------------------------------------------------


class TestTfidfClassifyReturnStructure:
    def setup_method(self):
        self.clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)

    def test_returns_dict(self):
        result = self.clf.classify("cells and DNA proteins")
        assert isinstance(result, dict)

    def test_has_predicted_category_key(self):
        result = self.clf.classify("cells and DNA proteins")
        assert "predicted_category" in result

    def test_has_scores_key(self):
        result = self.clf.classify("cells and DNA proteins")
        assert "scores" in result

    def test_scores_length_equals_number_of_journals(self):
        result = self.clf.classify("cells and DNA proteins")
        assert len(result["scores"]) == len(JOURNAL_DESCRIPTIONS)

    def test_each_score_entry_has_required_keys(self):
        result = self.clf.classify("cells and DNA proteins")
        for entry in result["scores"]:
            assert "label" in entry
            assert "score" in entry

    def test_scores_sorted_descending(self):
        result = self.clf.classify("quantum mechanics energy particle relativity")
        scores = [e["score"] for e in result["scores"]]
        assert scores == sorted(scores, reverse=True)

    def test_score_is_float(self):
        result = self.clf.classify("cells and DNA proteins")
        for entry in result["scores"]:
            assert isinstance(entry["score"], float)


# ---------------------------------------------------------------------------
# classify – scoring correctness
# ---------------------------------------------------------------------------


class TestTfidfClassifyScoring:
    def setup_method(self):
        self.clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)

    def test_biology_abstract_predicts_biology(self):
        abstract = "This paper investigates living organisms, cells, genetics, DNA and proteins."
        result = self.clf.classify(abstract)
        assert result["predicted_category"] == "biology"

    def test_physics_abstract_predicts_physics(self):
        abstract = "Analysis of quantum mechanics, particle interactions and energy levels under relativity."
        result = self.clf.classify(abstract)
        assert result["predicted_category"] == "physics"

    def test_chemistry_abstract_predicts_chemistry(self):
        abstract = "Synthesis of chemical compounds through catalytic molecular reactions and polymer formation."
        result = self.clf.classify(abstract)
        assert result["predicted_category"] == "chemistry"

    def test_no_overlap_returns_none(self):
        # Completely unrelated words that won't appear in vocabulary
        abstract = "zzz xxx yyy aaa bbb"
        result = self.clf.classify(abstract)
        assert result["predicted_category"] is None

    def test_scores_are_between_0_and_1(self):
        result = self.clf.classify("cells DNA quantum energy")
        for entry in result["scores"]:
            assert 0.0 <= entry["score"] <= 1.0

    def test_score_rounded_to_2_decimals(self):
        result = self.clf.classify("cells DNA quantum energy")
        for entry in result["scores"]:
            assert entry["score"] == round(entry["score"], 2)


# ---------------------------------------------------------------------------
# classify – single-journal edge case
# ---------------------------------------------------------------------------


class TestTfidfSingleJournal:
    def test_single_journal_matching_abstract(self):
        clf = TfidfClassifier(
            {"medicine": "Study of diseases, viruses, antibodies and vaccines."}
        )
        result = clf.classify(
            "Virus neutralisation by antibodies and vaccine efficacy."
        )
        assert result["predicted_category"] == "medicine"

    def test_single_journal_no_overlap_returns_none(self):
        clf = TfidfClassifier(
            {"medicine": "Study of diseases, viruses, antibodies and vaccines."}
        )
        result = clf.classify("zzz xxx yyy aaa bbb ccc")
        assert result["predicted_category"] is None


# ---------------------------------------------------------------------------
# get_top_matching_terms
# ---------------------------------------------------------------------------


class TestGetTopMatchingTerms:
    def setup_method(self):
        self.clf = TfidfClassifier(JOURNAL_DESCRIPTIONS)

    def test_returns_list(self):
        terms = self.clf.get_top_matching_terms("cells and DNA proteins genetics", "biology")
        assert isinstance(terms, list)

    def test_each_term_entry_has_required_keys(self):
        terms = self.clf.get_top_matching_terms("cells and DNA proteins genetics", "biology")
        for entry in terms:
            assert "term" in entry
            assert "score" in entry

    def test_respects_max_reasoning_terms_limit(self):
        clf = TfidfClassifier(JOURNAL_DESCRIPTIONS, max_reasoning_terms=2)
        terms = clf.get_top_matching_terms("cells and DNA proteins genetics living organisms", "biology")
        assert len(terms) <= 2

    def test_default_limit_is_5(self):
        terms = self.clf.get_top_matching_terms(
            "cells and DNA proteins genetics living organisms study", "biology"
        )
        assert len(terms) <= 5

    def test_terms_sorted_by_score_descending(self):
        terms = self.clf.get_top_matching_terms("cells and DNA proteins genetics", "biology")
        if len(terms) > 1:
            scores = [t["score"] for t in terms]
            assert scores == sorted(scores, reverse=True)

    def test_returns_empty_list_when_no_overlap(self):
        terms = self.clf.get_top_matching_terms("zzz xxx yyy aaa bbb", "biology")
        assert terms == []

    def test_term_scores_are_floats(self):
        terms = self.clf.get_top_matching_terms("cells and DNA proteins genetics", "biology")
        for entry in terms:
            assert isinstance(entry["score"], float)

    def test_invalid_predicted_category_raises(self):
        with pytest.raises(ValueError):
            self.clf.get_top_matching_terms("cells and DNA proteins genetics", "nonexistent_label")
