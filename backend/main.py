from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from ai.loader import load_document_from_path
from ai.chunking import process_document_to_chunks
from ai.scoring import calculate_all_scores
from ai.vector_index import create_hybrid_retriever, get_unique_retrieval_responses
from ai.llm import get_llm_response
from explain import classify_risk, explain_trust

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

origins=os.getenv('ORIGINS')

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins.split(",") if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded files
UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Store the path to the currently uploaded document
default_file_path: Optional[str] = None

@app.get("/")
def checking():
    return {"message": "TrustAI API is running"}


@app.post("/uploadpdf")
async def upload_doc(file: UploadFile = File(...)):
    """
    Upload a PDF document.
    The document will be saved and can be used later by evaluate_llm.
    """
    global default_file_path
    
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Save uploaded file persistently
        file_path = UPLOADS_DIR / file.filename
        content = await file.read()
        
        with open(file_path, 'wb') as saved_file:
            saved_file.write(content)
        
        # Store the path for later use by evaluate_llm
        default_file_path = str(file_path)
        
        # Return success response
        return JSONResponse(
            status_code=200,
            content={
                "message": "PDF uploaded successfully",
                "filename": file.filename,
                "file_path": default_file_path,
                "status": "ready_for_evaluation"
            }
        )       
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.post("/evaluate")
async def evaluate_llm(
    query: str = Query(..., description="The question to evaluate")
):
    """
    Evaluate LLM trustworthiness for a given query using the uploaded PDF.
    Performs: loading, chunking, vector indexing, LLM inference, scoring, and explanation.
    Requires a PDF to be uploaded first via /uploadpdf endpoint.
    """
    global default_file_path
    
    try:
        # Step 1: Load document from previously uploaded file
        if not default_file_path or not Path(default_file_path).exists():
            raise HTTPException(
                status_code=400, 
                detail="No document provided. Please upload a PDF first using /uploadpdf endpoint."
            )
        
        document_text = load_document_from_path(default_file_path)
        
        # Step 2: Process document to chunks
        chunks = process_document_to_chunks(document_text)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to create chunks from document")
        
        # Step 3: Create hybrid retriever and get retrieval responses
        hybrid_retriever = create_hybrid_retriever(chunks)
        unique_retrieval_responses = get_unique_retrieval_responses(hybrid_retriever, query, k=10)
        
        if not unique_retrieval_responses:
            raise HTTPException(
                status_code=400, 
                detail="No relevant evidence found for the query in the document"
            )
        
        # Step 4: Get LLM response
        llm_response = get_llm_response(hybrid_retriever, query)
        llm_answer = llm_response.get("answer", "")
        
        # Step 5: Calculate all scores
        evidence_chunks = [doc.page_content for doc in unique_retrieval_responses]
        scores = calculate_all_scores(llm_answer, evidence_chunks)
        
        # Step 6: Classify risk and get explanation
        risk_classification = classify_risk(scores["trust_score"])
        trust_explanation = explain_trust(scores["trust_score"])
        
        # Return results
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "llm_answer": llm_answer,
                "scores": scores,
                "risk_classification": risk_classification,
                "explanation": trust_explanation,
                "evidence_count": len(unique_retrieval_responses),
                "evidence_chunks": [
                    {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in unique_retrieval_responses[:5]
                ]
            }
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating LLM: {str(e)}")