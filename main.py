from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel
from typing import List
import os
import requests
from dotenv import load_dotenv

from my_module import load_documents, split_documents, build_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

app = FastAPI()

# A simple cache for storing vector stores
vector_store_cache = {}
embeddings = OpenAIEmbeddings()

# Define the request body model
class HackRxRequest(BaseModel):
    documents: str  # URL to a PDF
    questions: List[str]

@app.post("/hackrx/run")
async def run_hackrx(request: HackRxRequest, authorization: str = Header(...)):
    # Validate Authorization header
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    # Step 1: Check cache for existing vector store
    documents_url = request.documents
    if documents_url in vector_store_cache:
        vectorstore = vector_store_cache[documents_url]
    else:
        # Step 1: Download PDF (Only if not in cache)
        os.makedirs("SampleDocs", exist_ok=True)
        pdf_path = os.path.join("SampleDocs", "input.pdf")
        try:
            response = requests.get(documents_url)
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

        # Step 2: Process documents and build vectorstore
        try:
            docs = load_documents("SampleDocs/")
            chunks = split_documents(docs, embeddings)
            vectorstore = FAISS.from_documents(chunks, embeddings)
            # Cache the vectorstore
            vector_store_cache[documents_url] = vectorstore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    # Step 3: Answer the questions
    try:
        qa_chain = build_qa_chain(vectorstore)
        answers = []
        for q in request.questions:
            result = qa_chain({"query": q})
            answers.append(result["result"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")

    return {"answers": answers}