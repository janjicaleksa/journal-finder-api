import pytest
from app.services.keyword_classifier import KeywordMatcher


JOURNAL_KEYWORDS = {
    "biology": ["cell", "dna", "protein", "gene", "organism"],
    "physics": ["quantum", "particle", "energy", "wave", "relativity"],
    "chemistry": ["molecule", "reaction", "compound", "catalyst", "polymer"],
}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestKeywordMatcherInit:
    def test_valid_keywords_accepted(self):
        matcher = KeywordMatcher(JOURNAL_KEYWORDS)
        assert matcher.journal_keywords == JOURNAL_KEYWORDS

    def test_empty_dict_raises(self):
        with pytest.raises(ValueError, match="Journal keywords must not be empty"):
            KeywordMatcher({})

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            KeywordMatcher(None)


# ---------------------------------------------------------------------------
# classify – input validation
# ---------------------------------------------------------------------------


class TestClassifyInputValidation:
    def setup_method(self):
        self.matcher = KeywordMatcher(JOURNAL_KEYWORDS)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Abstract must not be empty"):
            self.matcher.classify("")

    def test_none_raises(self):
        with pytest.raises((ValueError, TypeError)):
            self.matcher.classify(None)


# ---------------------------------------------------------------------------
# classify – return structure
# ---------------------------------------------------------------------------


class TestClassifyReturnStructure:
    def setup_method(self):
        self.matcher = KeywordMatcher(JOURNAL_KEYWORDS)

    def test_returns_dict(self):
        result = self.matcher.classify("This paper studies cell dna protein gene organism.")
        assert isinstance(result, dict)

    def test_has_predicted_category_key(self):
        result = self.matcher.classify("cell dna")
        assert "predicted_category" in result

    def test_has_scores_key(self):
        result = self.matcher.classify("cell dna")
        assert "scores" in result

    def test_scores_length_equals_number_of_journals(self):
        result = self.matcher.classify("cell dna quantum")
        assert len(result["scores"]) == len(JOURNAL_KEYWORDS)

    def test_each_score_entry_has_required_keys(self):
        result = self.matcher.classify("cell dna")
        for entry in result["scores"]:
            assert "label" in entry
            assert "score" in entry
            assert "matched_keywords" in entry

    def test_scores_sorted_descending(self):
        result = self.matcher.classify("cell dna protein gene quantum particle")
        scores = [entry["score"] for entry in result["scores"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# classify – scoring correctness
# ---------------------------------------------------------------------------


class TestClassifyScoring:
    def setup_method(self):
        self.matcher = KeywordMatcher(JOURNAL_KEYWORDS)

    def test_perfect_match_gives_score_1(self):
        abstract = "cell dna protein gene organism"
        result = self.matcher.classify(abstract)
        biology_entry = next(e for e in result["scores"] if e["label"] == "biology")
        assert biology_entry["score"] == 1.0

    def test_partial_match_score_is_fraction(self):
        # 2 out of 5 biology keywords → 0.4
        abstract = "cell dna"
        result = self.matcher.classify(abstract)
        biology_entry = next(e for e in result["scores"] if e["label"] == "biology")
        assert biology_entry["score"] == pytest.approx(0.4)

    def test_zero_match_gives_score_0(self):
        abstract = "cell dna protein gene organism"
        result = self.matcher.classify(abstract)
        physics_entry = next(e for e in result["scores"] if e["label"] == "physics")
        assert physics_entry["score"] == 0.0

    def test_score_is_rounded_to_2_decimals(self):
        # 1 out of 3 chemistry keywords → 0.33...
        matcher = KeywordMatcher({"chem": ["molecule", "reaction", "compound"]})
        result = matcher.classify("molecule")
        entry = result["scores"][0]
        assert entry["score"] == pytest.approx(0.33)

    def test_matched_keywords_are_correct(self):
        abstract = "cell dna"
        result = self.matcher.classify(abstract)
        biology_entry = next(e for e in result["scores"] if e["label"] == "biology")
        assert set(biology_entry["matched_keywords"]) == {"cell", "dna"}

    def test_no_matched_keywords_when_score_is_zero(self):
        abstract = "cell dna protein gene organism"
        result = self.matcher.classify(abstract)
        physics_entry = next(e for e in result["scores"] if e["label"] == "physics")
        assert physics_entry["matched_keywords"] == []


# ---------------------------------------------------------------------------
# classify – predicted_category
# ---------------------------------------------------------------------------


class TestClassifyPredictedCategory:
    def setup_method(self):
        self.matcher = KeywordMatcher(JOURNAL_KEYWORDS)

    def test_best_matching_label_returned(self):
        # biology gets 5/5, others get 0
        abstract = "cell dna protein gene organism"
        result = self.matcher.classify(abstract)
        assert result["predicted_category"] == "biology"

    def test_no_match_returns_none(self):
        abstract = "this abstract contains no relevant terms whatsoever"
        result = self.matcher.classify(abstract)
        assert result["predicted_category"] is None

    def test_case_insensitive_matching(self):
        abstract = "CELL DNA PROTEIN GENE ORGANISM"
        result = self.matcher.classify(abstract)
        assert result["predicted_category"] == "biology"

    def test_keyword_embedded_in_word_still_matches(self):
        # "gene" is contained in "generated"
        abstract = "generated"
        result = self.matcher.classify(abstract)
        biology_entry = next(e for e in result["scores"] if e["label"] == "biology")
        assert "gene" in biology_entry["matched_keywords"]

    def test_tie_winner_is_first_after_sort_stability(self):
        # Both journals have 1 keyword each → score 1.0
        matcher = KeywordMatcher({
            "a": ["alpha"],
            "b": ["beta"],
        })
        result = matcher.classify("alpha beta")
        # Both score 1.0; the top entry must be one of the two labels
        assert result["predicted_category"] in ("a", "b")
        assert result["scores"][0]["score"] == 1.0


# ---------------------------------------------------------------------------
# classify – single-journal edge case
# ---------------------------------------------------------------------------


class TestSingleJournal:
    def test_single_journal_full_match(self):
        matcher = KeywordMatcher({"medicine": ["virus", "antibody", "vaccine"]})
        result = matcher.classify("The virus was neutralised by the antibody and vaccine.")
        assert result["predicted_category"] == "medicine"
        assert result["scores"][0]["score"] == 1.0

    def test_single_journal_no_match(self):
        matcher = KeywordMatcher({"medicine": ["virus", "antibody", "vaccine"]})
        result = matcher.classify("quantum relativity energy wave")
        assert result["predicted_category"] is None
        assert result["scores"][0]["score"] == 0.0
