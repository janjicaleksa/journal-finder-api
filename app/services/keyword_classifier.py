class KeywordMatcher:
    """Classify a paper abstract into a journal category using keyword matching method."""
    def __init__(self, journal_keywords: dict[str, list[str]]):
        self.journal_keywords = journal_keywords

    def classify(self, abstract: str):
        text = abstract.lower()

        if not text:
            raise ValueError("Abstract must not be empty.")

        results = []

        for label, keywords in self.journal_keywords.items():
            matched = []

            for keyword in keywords:
                if keyword in text:
                    matched.append(keyword)

            score = len(matched) / len(keywords)

            results.append({
                "label": label,
                "score": round(score, 2),
                "matched_keywords": matched
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        return results