from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from vector_index import chunks
from llm import llm_response

class RetrievalOverlapScorer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def score(self, llm_response: str, evidence_chunks: List[str]) -> float:
         #llm_response: the LLM-generated response, evidence_chunks: retrieved document chunks
             
        if not llm_response or not evidence_chunks:
            return 0.0

        answer_sentences = self._split_sentences(llm_response)

        answer_emb = self.model.encode(answer_sentences, normalize_embeddings=True)
        evidence_emb = self.model.encode(evidence_chunks, normalize_embeddings=True)

        similarities = cosine_similarity(answer_emb, evidence_emb)

        # For each answer sentence, take max similarity to evidence
        max_scores = similarities.max(axis=1)

        return float(np.mean(max_scores))

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in text.split(".") if s.strip()]


retreival_scorer = RetrievalOverlapScorer()

retreival_score = retreival_scorer.score(
    llm_response=llm_response["answer"],
    evidence_chunks=[doc.page_content for doc in chunks]
)

print("Retrieval overlap:", retreival_score)


class SemanticSimilarityScorer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def score(self, llm_response: str, evidence_chunks: List[str]) -> float:
        if not llm_response or not evidence_chunks:
            return 0.0

        # Embed full answer
        answer_embedding = self.model.encode([llm_response], normalize_embeddings=True)
        # Embed all evidence chunks
        evidence_embeddings = self.model.encode(evidence_chunks, normalize_embeddings=True)

        # Compare answer to all evidence
        similarities = cosine_similarity(answer_embedding, evidence_embeddings)

        # Use max or mean (max is safer for RAG)
        return float(similarities.max())

semantic_scorer = RetrievalOverlapScorer()

semantic_score = semantic_scorer.score(
    llm_response=llm_response["answer"],
    evidence_chunks=[doc.page_content for doc in chunks]
)

print("semantic_score:", semantic_score)
