from fastapi import FastAPI, UploadFile, File
import shutil, os
from database import SessionLocal
from models import Document
from worker import process_document
from agent import route_query
from storage import save_file

from pydantic import BaseModel
from processing import extract_text, chunk_text, store_chunks, answer_question

app = FastAPI(title="Agentic RAG Platform")

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    save_file(file.file, file.filename)

    db = SessionLocal()
    new_doc = Document(filename=file.filename, status="pending")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    document_id = new_doc.id
    db.close()

    process_document.delay(document_id)

    return {"document_id": document_id, "filename": file.filename, "status": "pending"}

class Question(BaseModel):
    question: str
    document_id: int | None = None

@app.post("/ask")
async def ask_question(payload: Question):
    result = route_query(payload.question, payload.document_id)
    return result