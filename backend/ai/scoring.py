from typing import List, Any
import numpy as np
import re
import transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch


from vector_index import retrieval_responses
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
    evidence_chunks=[doc.page_content for doc in retrieval_responses]
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

semantic_scorer = SemanticSimilarityScorer()

semantic_score = semantic_scorer.score(
    llm_response=llm_response["answer"],
    evidence_chunks=[doc.page_content for doc in retrieval_responses]
)

print("semantic_score:", semantic_score)


# Suppress the warning about unused weights
transformers.logging.set_verbosity_error()

class NLIScorer:
    def __init__(self, model_name="roberta-large-mnli", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Label mapping for roberta-large-mnli
        self.label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    def _predict(self, premise: str, hypothesis: str):
        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2])
        }

    def score(self, llm_response: str, evidence_chunks: List[str]) -> dict:
        """
        Returns NLI risk scores:
        - contradiction_risk
        - neutral_score
        - entailment_score
        """

        if not llm_response or not evidence_chunks:
            return {
                "contradiction_risk": 1.0,
                "neutral_score": 0.0,
                "entailment_score": 0.0
            }

        answer_sentences = self.split_sentences(llm_response)

        contradiction_scores = []
        neutral_scores = []
        entailment_scores = []

        for ans_sent in answer_sentences:
            sent_contradictions = []
            sent_neutrals = []
            sent_entailments = []

            for chunk in evidence_chunks:
                result = self._predict(premise=chunk, hypothesis=ans_sent)
                sent_contradictions.append(result["contradiction"])
                sent_neutrals.append(result["neutral"])
                sent_entailments.append(result["entailment"])

            # WORST-CASE logic
            contradiction_scores.append(max(sent_contradictions))
            neutral_scores.append(max(sent_neutrals))
            entailment_scores.append(max(sent_entailments))

        return {
            "contradiction_risk": float(np.mean(contradiction_scores)),
            "neutral_score": float(np.mean(neutral_scores)),
            "entailment_score": float(np.mean(entailment_scores))
        }


NLIScorer = NLIScorer()

NLI_Score = NLIScorer.score(
    llm_response=llm_response["answer"],
    evidence_chunks=[doc.page_content for doc in retrieval_responses]
)

print("NLI Score:")
print(f"  Contradiction Risk: {NLI_Score['contradiction_risk']:.4f}")
print(f"  Neutral Score: {NLI_Score['neutral_score']:.4f}")
print(f"  Entailment Score: {NLI_Score['entailment_score']:.4f}")

trust_score = (
    0.4 * retreival_score +
    0.4 * semantic_score +
    0.2 * (1 - NLI_Score['contradiction_risk'])
) * 100
print(f"Trust score: {trust_score}")