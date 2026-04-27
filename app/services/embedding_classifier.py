import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingClassifier:
    """Classify a paper abstract into a journal category using sentence embeddings."""

    def __init__(self, journal_descriptions: dict[str, str], embedding_model_name: str):
        """Initialization of the Embedding Classifier.

        Notes: - The model is loaded and journal embeddings are computed only once during
                 service initialization, not on every classify call.
               - Cosine similarity between the abstract embedding and each pre-computed
                 journal description embedding is used to rank the journals.
               - Journal descriptions are the same predefined descriptions used by the TF-IDF classifier.
        """
        if not journal_descriptions:
            raise ValueError("Journal descriptions must not be empty.")

        self.journal_labels = list(journal_descriptions.keys())
        self.journal_texts = list(journal_descriptions.values())
        self.embedding_model_name = embedding_model_name

        self.model = SentenceTransformer(self.embedding_model_name)
        self.journal_embeddings = self.model.encode(
            self.journal_texts, normalize_embeddings=True
        )

    def classify(self, abstract: str) -> dict:
        """Classify a paper abstract and return the predicted journal category, all scores and the abstract embedding.

        Returns a dict with keys:
            - predicted_category: label of the best-matching journal, or None if all scores are zero.
            - scores: list of {"label": str, "score": float} dicts sorted by score in descending order.
        """
        if not abstract:
            raise ValueError("Abstract must not be empty.")

        abstract_embedding = self.model.encode([abstract], normalize_embeddings=True)
        # similarities = cosine_similarity(abstract_embedding, self.journal_embeddings)[0]
        similarities = np.dot(abstract_embedding, self.journal_embeddings.T).flatten()

        scores = []
        for label, score in zip(self.journal_labels, similarities):
            scores.append({"label": label, "score": round(float(score), 2)})

        scores.sort(key=lambda x: x["score"], reverse=True)

        predicted_category = scores[0]["label"] if scores[0]["score"] > 0.0 else None

        return {"predicted_category": predicted_category, "scores": scores}

    def get_top_supporting_sentences(
        self, abstract: str, predicted_category: str, top_k: int = 2
    ):
        """Get the top supporting sentences for the predicted journal category.

        Notes: - Suporting sentences are the sentences that are most similar to the predicted journal category.
               - This method works well for larger abstracts.
        """

        sentences = []
        for sentence in re.split(r"(?<=[.!?])\s+", abstract.strip()):
            if sentence.strip():
                sentences.append(sentence.strip())

        if not sentences:
            return []

        best_class_index = self.journal_labels.index(predicted_category)
        best_class_embedding = self.journal_embeddings[best_class_index]

        sentences_embeddings = self.model.encode(sentences, normalize_embeddings=True)
        scores = np.dot(sentences_embeddings, best_class_embedding)

        sentence_scores = []
        for i, sentence in enumerate(sentences):
            sentence_scores.append(
                {"sentence": sentence, "score": round(float(scores[i]), 2)}
            )

        sentence_scores.sort(key=lambda x: x["score"], reverse=True)

        return sentence_scores[:top_k]
