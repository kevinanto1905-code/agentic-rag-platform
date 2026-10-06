from processing import answer_question, summarize_document

def route_query(query: str, document_id: int = None) -> dict:
    query_lower = query.lower()

    if "summarize" in query_lower or "summary" in query_lower:
        if document_id is None:
            return {"answer": "Please specify which document to summarize.", "tool_used": "none"}
        summary = summarize_document(document_id)
        return {"answer": summary, "tool_used": "summarize_tool"}

    else:
        result = answer_question(query, document_id=document_id)
        result["tool_used"] = "retrieval_tool"
        return result