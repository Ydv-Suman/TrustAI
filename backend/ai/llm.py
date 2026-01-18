from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

import os
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """
You are an assistant answering questions strictly based on the provided evidence.
You MUST NOT use outside knowledge.
If the answer is not fully supported by the evidence, respond with:
"Insufficient evidence to answer the question."

Evidence:
{context}

Question:
{input}

Answer:
"""


def build_rag_chain(hybrid_retriever):
    """Build a RAG chain with the given hybrid retriever."""
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0
    )

    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=SYSTEM_PROMPT
    )

    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=hybrid_retriever,
        combine_docs_chain=document_chain
    )

    return rag_chain


def get_llm_response(hybrid_retriever, query: str):
    """Get LLM response using RAG chain."""
    rag_chain = build_rag_chain(hybrid_retriever)
    llm_response = rag_chain.invoke({"input": query})
    return llm_response