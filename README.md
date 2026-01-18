# TrustAI

TrustAI is a comprehensive tool for evaluating the trustworthiness of Large Language Model (LLM) responses by analyzing their alignment with source documents. The system uses multiple scoring mechanisms including retrieval overlap, semantic similarity, and Natural Language Inference (NLI) to assess whether an LLM's answer is grounded in evidence or contains potential hallucinations.

## Features

- **PDF Document Upload**: Upload PDF documents as source material for evaluation
- **Query Evaluation**: Submit queries and receive LLM-generated answers
- **Multi-dimensional Scoring**:
  - **Retrieval Overlap Score**: Measures how well sentences in the LLM answer align with retrieved evidence chunks
  - **Semantic Similarity Score**: Evaluates overall semantic alignment between the answer and evidence
  - **Natural Language Inference (NLI)**: Detects contradictions, neutral content, and entailment using a RoBERTa-based model
- **Trust Score Calculation**: Combines multiple metrics into a single trust score (0-100)
- **Risk Classification**: Automatically classifies responses as "High Trust", "Medium Trust", or "High Hallucination Risk"
- **Evidence Display**: Shows retrieved evidence chunks that support the evaluation
- **Hybrid Retrieval**: Uses both semantic (FAISS) and keyword-based (BM25) retrieval for comprehensive evidence gathering

## Tech Stack

### Backend

- **FastAPI**: Web framework for building the API
- **LangChain**: Framework for LLM integration and document processing
- **OpenAI**: LLM provider for generating answers
- **FAISS**: Vector database for semantic similarity search
- **Sentence Transformers**: Embedding models for semantic similarity
- **Transformers (Hugging Face)**: RoBERTa model for NLI scoring
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities

### Frontend

- **React 19**: UI framework
- **Vite**: Build tool and development server

### Project Structure

```
TrustAI/
├── backend/
│   ├── ai/
│   │   ├── chunking.py          # Document chunking logic
│   │   ├── loader.py             # PDF document loading
│   │   ├── llm.py                # LLM integration and query handling
│   │   ├── scoring.py            # Trust scoring algorithms
│   │   └── vector_index.py       # Hybrid retrieval (FAISS + BM25)
│   ├── data/                     # Sample documents
│   ├── uploads/                  # Uploaded PDF files storage
│   ├── explain.py                # Risk classification and explanations
│   ├── main.py                   # FastAPI application and endpoints
│   └── requirements.txt          # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   ├── App.css               # Application styles
│   │   ├── main.jsx              # React entry point
│   │   └── index.css             # Global styles
│   ├── package.json              # Node.js dependencies
│   └── vite.config.js            # Vite configuration
│
└── README.md                     # This file
```

## Docker Images

Pull the pre-built Docker images:

```bash
docker pull sumanydv/trustai-backend
docker pull sumanydv/trustai-frontend
```

## How It Works

1. **Document Upload**: User uploads a PDF document that serves as the knowledge source
2. **Document Processing**: The PDF is loaded, chunked into smaller pieces, and indexed using both semantic (FAISS) and keyword-based (BM25) methods
3. **Query Processing**: When a query is submitted:
   - The system retrieves relevant evidence chunks using hybrid retrieval
   - An LLM (via OpenAI) generates an answer based on the retrieved context
   - Multiple scoring mechanisms evaluate the answer against the evidence
4. **Trust Evaluation**:
   - **Retrieval Overlap**: Compares individual sentences in the answer with evidence chunks
   - **Semantic Similarity**: Measures overall semantic alignment
   - **NLI Scoring**: Detects contradictions, neutral statements, and entailment relationships
5. **Risk Classification**: The combined trust score determines risk level:
   - **High Trust** (≥85): Strongly supported by evidence
   - **Medium Trust** (70-84): Mostly supported, caution advised
   - **High Hallucination Risk** (<70): Weak evidence support

## Environment Variables

### Backend (.env)

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ORIGINS`: Comma-separated list of allowed CORS origins (default: allows all)

### Frontend (.env)

- `VITE_API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
