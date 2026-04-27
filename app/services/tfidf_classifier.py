import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfClassifier:
    """Classify a paper abstract into a journal category using TF-IDF method."""

    def __init__(
        self,
        journal_descriptions: dict[str, str],
        ngram_range: tuple[int, int] = (1, 2),
        max_reasoning_terms: int = 5,
    ):
        """Initializazion of the TF-IDF classifier

        Notes: - The vectorizer is fit only once during service initialization so that it is not re-fit for each abstract.
               - N-grams is used to tokenize the text into n-grams (in single or two consecutive words - unigrams and bigrams).
               - Max reasoning terms is the maximum number (set on 5 by default) of terms to return for reasoning.
               - Journal descriptions are the predefined descriptions, same as the journal keywords in the keyword matching classifier.
        """
        if not journal_descriptions:
            raise ValueError("Journal descriptions must not be empty.")

        self.journal_labels = list(journal_descriptions.keys())
        self.journal_texts = list(journal_descriptions.values())
        self.ngram_range = ngram_range
        self.max_reasoning_terms = max_reasoning_terms

        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=ngram_range
        )
        self.journal_vectors = self.vectorizer.fit_transform(self.journal_texts)

    def classify(self, abstract: str) -> dict:
        """Classify a paper abstract and return the predicted journal category and all scores.

        Returns a dict with keys:
            - predicted_category: label of the best-matching journal, or None if all scores are zero.
            - scores: list of {"label": str, "score": float} dicts sorted by score in descending order.
        """

        if not abstract:
            raise ValueError("Abstract must not be empty.")

        abstract_vector = self.vectorizer.transform([abstract.lower()])
        similarities = cosine_similarity(abstract_vector, self.journal_vectors)[0]

        scores = []
        for label, score in zip(self.journal_labels, similarities):
            scores.append({"label": label, "score": round(float(score), 2)})

        scores.sort(key=lambda x: x["score"], reverse=True)

        predicted_category = scores[0]["label"] if scores[0]["score"] > 0.0 else None

        return {
            "predicted_category": predicted_category,
            "scores": scores,
        }

    def get_top_matching_terms(
        self, abstract: str, predicted_category: str
    ) -> list[dict[str, float | str]]:
        """Get the top matching terms between the abstract and the predicted journal."""

        abstract_vector = self.vectorizer.transform([abstract.lower()])
        best_journal_index = self.journal_labels.index(predicted_category)
        best_journal_vector = self.journal_vectors[best_journal_index]

        overlap = abstract_vector.multiply(best_journal_vector)
        feature_names = np.asarray(self.vectorizer.get_feature_names_out())
        non_zero_indices = overlap.nonzero()[1]

        term_scores = []
        for index in non_zero_indices:
            term_scores.append(
                {
                    "term": str(feature_names[index]),
                    "score": round(float(overlap[0, index]), 2),
                }
            )

        term_scores.sort(key=lambda x: x["score"], reverse=True)

        return term_scores[: self.max_reasoning_terms]
