class KeywordMatcher:
    """Classify a paper abstract into a journal category using keyword matching method."""

    def __init__(self, journal_keywords: dict[str, list[str]]):
        """Initialization of the Keyword Matcher

        Notes: - Journal keywords are the predefined keywords that are used to classify the papers into the journal topics.
        """
        if not journal_keywords:
            raise ValueError("Journal keywords must not be empty.")

        self.journal_keywords = journal_keywords

    def classify(self, abstract: str) -> dict:
        """Classify a paper abstract and return the predicted journal category, all scores and the matched keywords.

        Returns a dict with keys:
            - predicted_category: label of the best-matching journal, or None if no keywords matched.
            - scores: list of {"label": str, "score": float, "matched_keywords": list} dicts sorted by score descending.
        """
        if not abstract:
            raise ValueError("Abstract must not be empty.")

        scores = []

        for label, keywords in self.journal_keywords.items():
            matched = []

            for keyword in keywords:
                if keyword in abstract.lower():
                    matched.append(keyword)

            score = len(matched) / len(keywords)

            scores.append(
                {"label": label, "score": round(score, 2), "matched_keywords": matched}
            )

        scores.sort(key=lambda x: x["score"], reverse=True)

        predicted_category = scores[0]["label"] if scores[0]["score"] > 0.0 else None

        return {
            "predicted_category": predicted_category,
            "scores": scores,
        }
