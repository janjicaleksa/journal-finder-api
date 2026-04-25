from app.services.keyword_classifier import KeywordMatcher
from app.services.tfidf_classifier import TfidfClassifier
from app.services.embedding_classifier import EmbeddingClassifier

class CompareService:
    """Run all three classifiers on the same abstract and return a side-by-side comparison."""

    def __init__(
        self,
        keyword_matcher: KeywordMatcher,
        tfidf_classifier: TfidfClassifier,
        embedding_classifier: EmbeddingClassifier,
    ):
        self.keyword_matcher = keyword_matcher
        self.tfidf_classifier = tfidf_classifier
        self.embedding_classifier = embedding_classifier

    def compare(self, abstract: str) -> dict:
        """Run all three classifiers and return their top prediction and score.

        Returns a dict with keys 'keyword_matching', 'tfidf', and 'embedding'.
        Each value is a dict with:
            - predicted_category: label of the best-matching journal or None.
            - score: cosine / keyword score of the predicted category or None.
        """
        if not abstract:
            raise ValueError("Abstract must not be empty.")

        keyword_matcher_result = self.keyword_matcher.classify(abstract)
        tfidf_result = self.tfidf_classifier.classify(abstract)
        embedding_result = self.embedding_classifier.classify(abstract)

        keyword_predicted_category = keyword_matcher_result["predicted_category"]
        if keyword_matcher_result["scores"] and keyword_matcher_result["scores"][0]["score"] > 0.0:
            keyword_score = keyword_matcher_result["scores"][0]["score"]
        else:
            keyword_score = None
            
        tfidf_predicted_category = tfidf_result["predicted_category"]
        if tfidf_result["scores"] and tfidf_result["scores"][0]["score"] > 0.0:
            tfidf_score = tfidf_result["scores"][0]["score"]
        else:
            tfidf_score = None

        embedding_predicted_category = embedding_result["predicted_category"]
        if embedding_result["scores"] and embedding_result["scores"][0]["score"] > 0.0:
            embedding_score = embedding_result["scores"][0]["score"]
        else:
            embedding_score = None

        return {
            "keyword_matching": {
                "predicted_category": keyword_predicted_category,
                "score": keyword_score,
            },
            "tfidf": {
                "predicted_category": tfidf_predicted_category,
                "score": tfidf_score,
            },
            "embedding": {
                "predicted_category": embedding_predicted_category,
                "score": embedding_score,
            },
        }