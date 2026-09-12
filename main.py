from fastapi import FastAPI, UploadFile, File
import shutil, os
from database import SessionLocal
from models import Document
from worker import process_document

app = FastAPI(title="Agentic RAG Platform")

@app.get("/")
def health_check():
    return {"status": "running"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)
    save_path = f"uploads/{file.filename}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db = SessionLocal()
    new_doc = Document(filename=file.filename, status="pending")
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    document_id = new_doc.id
    db.close()

    process_document.delay(document_id)

    return {"document_id": document_id, "filename": file.filename, "status": "pending"}