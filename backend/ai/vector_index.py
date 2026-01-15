from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from typing import List
import numpy as np

from chunking import chunks


# Initialize embedding model
# Using sentence-transformers model optimized for semantic similarity
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


# Create FAISS vector store from document chunks
# This loads the chunks created by the chunking module and builds a searchable index
dense_vectorstore = FAISS.from_documents(chunks, embeddings)
dense_retriever = dense_vectorstore.as_retriever()

sparse_retriever = BM25Retriever.from_documents(chunks)
sparse_retriever.k = 3

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weight=[0.7,0.3]
)

hybrid_retriever


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


query = str(input("Enter your query: "))
retrieval_responses = hybrid_retriever.invoke(query, k=10)

print(f"\nQuery: {query}")
print(f"Results before deduplication: {len(retrieval_responses)}")

unique_retrieval_responses = deduplicate_results(retrieval_responses)
print(f"Number of unique matches after deduplication: {len(unique_retrieval_responses)}\n")

for idx, response in enumerate(unique_retrieval_responses, start=1):
    print(f"Match {idx}:")
    print(f"Content `vector_index response`: {response.page_content}")
    print(f"Metadata: {response.metadata}\n")