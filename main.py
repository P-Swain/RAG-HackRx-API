from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import hashlib

# Import the updated offline function
from my_module import search_pinecone_memory_offline

# Load environment variables
load_dotenv()

app = FastAPI()

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    # Namespace for document chunks
    doc_namespace = hashlib.sha256(request.documents.encode()).hexdigest()
    # Separate namespace for Q&A cache
    qa_namespace = f"{doc_namespace}-qa"

    answers = []
    
    # --- OFFLINE LOGIC ---
    # Fetch all cached answers for this document namespace at once
    try:
        all_cached_qa = await run_in_threadpool(search_pinecone_memory_offline, qa_namespace)
        
        for q in request.questions:
            # Check for an exact match in the fetched cache
            answer = all_cached_qa.get(q)
            
            if answer:
                answers.append({
                    "question": q,
                    "answer": answer,
                    "from_memory": True
                })
            else:
                # If not in cache, append a message indicating it, as OpenAI is offline
                answers.append({
                    "question": q,
                    "answer": "Answer not found in cache. Cannot generate new answer as OpenAI key is not functional.",
                    "from_memory": False
                })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data from Pinecone: {str(e)}")

    return {
        "answers": answers
    }