from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from typing import List
import numpy as np


# Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def embedded_text(texts: List[str]) -> np.ndarray:
    """Embed a list of text strings into a numpy array using HuggingFace embeddings."""
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    # Use the embeddings model to embed all texts
    embeddings_list = embeddings.embed_documents(texts)
    
    # Convert to numpy array for easier manipulation
    return np.array(embeddings_list)


def create_hybrid_retriever(chunks: List[Document]):
    """Create a hybrid retriever from document chunks."""
    # Create FAISS vector store from document chunks
    dense_vectorstore = FAISS.from_documents(chunks, embeddings)
    dense_retriever = dense_vectorstore.as_retriever()

    # Create BM25 sparse retriever
    sparse_retriever = BM25Retriever.from_documents(chunks)
    sparse_retriever.k = 3

    # Create ensemble retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weight=[0.7, 0.3]
    )

    return hybrid_retriever


def deduplicate_results(results):
    """Remove duplicate Document objects based on their page_content."""
    seen = set()
    unique = []

    for doc in results:
        # Normalize content for comparison (remove extra whitespace, lowercase)
        normalized = " ".join(doc.page_content.split()).lower()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(doc)

    return unique


def get_unique_retrieval_responses(hybrid_retriever, query: str, k: int = 10):
    """Get unique retrieval responses for a query using the hybrid retriever."""
    retrieval_responses = hybrid_retriever.invoke(query, k=k)
    unique_retrieval_responses = deduplicate_results(retrieval_responses)
    return unique_retrieval_responses
