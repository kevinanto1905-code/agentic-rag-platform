from pypdf import PdfReader
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-documents")


def extract_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks


def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def store_chunks(document_id: int, chunks: list[str]):
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"doc{document_id}-chunk{i}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "document_id": document_id,
                "chunk_index": i
            }
        })
    index.upsert(vectors=vectors)
    return len(vectors)
def semantic_search(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = get_embedding(query)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    matches = []
    for match in results["matches"]:
        matches.append({
            "text": match["metadata"]["text"],
            "document_id": match["metadata"]["document_id"],
            "score": match["score"]
        })
    return matches


def answer_question(query: str, top_k: int = 5) -> dict:
    matches = semantic_search(query, top_k=top_k)

    if not matches:
        return {
            "answer": "I don't have any documents to search yet. Please upload one first.",
            "sources": []
        }

    context = "\n\n".join([f"[Chunk {i}]: {m['text']}" for i, m in enumerate(matches)])

    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": matches
    }