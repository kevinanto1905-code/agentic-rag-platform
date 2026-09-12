from celery import Celery
from database import SessionLocal
from models import Document
from processing import extract_text, chunk_text, store_chunks

celery_app = Celery(
    "tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_document(document_id: int):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            print(f"Document {document_id} not found")
            return

        doc.status = "processing"
        db.commit()
        print(f"Document {document_id} -> processing")

        file_path = f"uploads/{doc.filename}"
        text = extract_text(file_path)
        chunks = chunk_text(text)
        print(f"Extracted {len(text)} characters, split into {len(chunks)} chunks")

        stored_count = store_chunks(document_id, chunks)
        print(f"Stored {stored_count} embeddings in Pinecone")

        doc.status = "ready"
        db.commit()
        print(f"Document {document_id} -> ready")

    except Exception as e:
        doc.status = "failed"
        db.commit()
        print(f"Document {document_id} -> failed: {e}")
    finally:
        db.close()