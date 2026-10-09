import json
import pandas as pd
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


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 20) -> list[str]:
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
MIN_SCORE = 0.2  # chunks scoring below this are treated as "not relevant"


def semantic_search(query: str, top_k: int = 3, document_id: int | None = None) -> list[dict]:
    query_embedding = get_embedding(query)
    search_args = dict(vector=query_embedding, top_k=top_k * 3, include_metadata=True)
    if document_id is not None:
        search_args["filter"] = {"document_id": {"$eq": document_id}}
    results = index.query(**search_args)

    matches, seen_texts = [], set()
    for match in results["matches"]:
        text = match["metadata"]["text"]
        if match["score"] < MIN_SCORE or text in seen_texts:
            continue
        seen_texts.add(text)
        matches.append({
            "text": text,
            "document_id": match["metadata"]["document_id"],
            "score": round(match["score"], 3),
        })
        if len(matches) == top_k:
            break
    return matches


def answer_question(query: str, top_k: int = 3, document_id: int | None = None) -> dict:
    matches = semantic_search(query, top_k=top_k, document_id=document_id)

    if not matches:
        return {"answer": "I don't have enough information to answer that.", "sources": []}

    context = "\n\n".join(f"[Source {i + 1}]: {m['text']}" for i, m in enumerate(matches))

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "You answer questions using ONLY the provided context. "
                "If the answer is not in the context, reply exactly: "
                "I don't have enough information to answer that. "
                "Keep answers short and mention which source you used, e.g. (Source 2)."
            )},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )

    sources = [
        {"source": i + 1, "document_id": m["document_id"], "score": m["score"],
         "preview": m["text"][:150] + "..."}
        for i, m in enumerate(matches)
    ]
    return {"answer": response.choices[0].message.content, "sources": sources}


def summarize_document(document_id: int) -> str:
    query_embedding = get_embedding("overview and main topics of this document")
    results = index.query(
        vector=query_embedding,
        top_k=50,
        include_metadata=True,
        filter={"document_id": {"$eq": document_id}},
    )
    matches = sorted(results["matches"], key=lambda m: m["metadata"]["chunk_index"])
    if not matches:
        return "No content found for that document ID."

    full_text = "\n\n".join(m["metadata"]["text"] for m in matches)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content":
            f"Summarize the following document in 4-5 clear sentences covering its main topics.\n\n{full_text}"}],
    )
    return response.choices[0].message.content

ALLOWED_OPS = {"sum", "mean", "median", "min", "max", "count"}


def load_spreadsheet(file_path: str) -> pd.DataFrame:
    if file_path.lower().endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _clean(value):
    value = float(value)
    return int(value) if value.is_integer() else round(value, 2)


def analyze_spreadsheet(file_path: str, question: str) -> dict:
    df = load_spreadsheet(file_path)
    schema = ", ".join(f"{name} ({dtype})" for name, dtype in df.dtypes.astype(str).items())
    sample = df.head(3).to_string(index=False)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "You turn a question about a spreadsheet into a JSON plan. "
                "Reply with JSON only, using exactly these keys: "
                '"operation" (one of sum, mean, median, min, max, count), '
                '"column" (column to calculate on, or null when counting rows), '
                '"group_by" (column to group by, or null), '
                '"filter_column" (column to filter on, or null), '
                '"filter_value" (value to match, or null). '
                "Use only column names that exist in the schema."
            )},
            {"role": "user", "content": f"Columns: {schema}\nFirst rows:\n{sample}\n\nQuestion: {question}"},
        ],
    )

    try:
        plan = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"answer": "I couldn't turn that question into a calculation.", "plan": None}

    op = plan.get("operation")
    col = plan.get("column")
    group_by = plan.get("group_by")
    f_col = plan.get("filter_column")
    f_val = plan.get("filter_value")

    if op not in ALLOWED_OPS:
        return {"answer": "That kind of calculation isn't supported yet.", "plan": plan}
    for name in (col, group_by, f_col):
        if name is not None and name not in df.columns:
            return {"answer": f"I couldn't find a column named '{name}'.", "plan": plan}

    if f_col is not None:
        df = df[df[f_col].astype(str).str.lower() == str(f_val).lower()]
        if df.empty:
            return {"answer": "No rows matched that filter.", "plan": plan}

    if col is None:
        if op != "count":
            return {"answer": "Please tell me which column to calculate on.", "plan": plan}
        result = df.groupby(group_by).size() if group_by else len(df)
    else:
        if op != "count" and not pd.api.types.is_numeric_dtype(df[col]):
            return {"answer": f"'{col}' isn't a numeric column, so I can't {op} it.", "plan": plan}
        data = df.groupby(group_by)[col] if group_by else df[col]
        result = getattr(data, op)()

    if isinstance(result, pd.Series):
        value = {str(k): _clean(v) for k, v in result.items()}
    else:
        value = _clean(result)

    return {"answer": f"{op} of {col or 'rows'} = {value}", "result": value, "plan": plan}