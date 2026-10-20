import os
from celery import Celery
from database import SessionLocal
from models import Document
from processing import extract_text, chunk_text, store_chunks, load_spreadsheet
from storage import get_local_path

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
@celery_app.task
def process_document(document_id: int):
    db = SessionLocal()
    doc = None
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            print(f"Document {document_id} not found")
            return

        doc.status = "processing"
        db.commit()
        print(f"Document {document_id} -> processing")

        file_path = get_local_path(doc.filename)

        if doc.filename.lower().endswith((".csv", ".xlsx")):
            df = load_spreadsheet(file_path)
            print(f"Spreadsheet loaded: {len(df)} rows, {len(df.columns)} columns")
        else:
            text = extract_text(file_path)
            chunks = chunk_text(text)
            print(f"Extracted {len(text)} characters, split into {len(chunks)} chunks")
            stored_count = store_chunks(document_id, chunks)
            print(f"Stored {stored_count} embeddings in Pinecone")

        doc.status = "ready"
        db.commit()
        print(f"Document {document_id} -> ready")

    except Exception as e:
        if doc:
            doc.status = "failed"
            db.commit()
        print(f"Document {document_id} -> failed: {e}")
    finally:
        db.close()