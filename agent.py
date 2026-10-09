from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from processing import answer_question, summarize_document, analyze_spreadsheet, openai_client
from database import SessionLocal
from models import Document


class AgentState(TypedDict):
    query: str
    document_id: Optional[int]
    route: str
    result: dict


def router_node(state: AgentState) -> dict:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "Classify the user's request. Reply with exactly one word: "
                "'summarize' if they want an overview or summary of a whole document; "
                "'analyze' if they want a calculation on spreadsheet data "
                "(totals, averages, counts, highest or lowest, grouped by a column); "
                "otherwise 'retrieval' (a specific question about document text)."
            )},
            {"role": "user", "content": state["query"]},
        ],
    )
    choice = response.choices[0].message.content.strip().lower()
    if "summarize" in choice:
        route = "summarize"
    elif "analyze" in choice:
        route = "analyze"
    else:
        route = "retrieval"
    return {"route": route}


def retrieval_node(state: AgentState) -> dict:
    result = answer_question(state["query"], document_id=state["document_id"])
    result["tool_used"] = "retrieval_tool"
    return {"result": result}


def summarize_node(state: AgentState) -> dict:
    if state["document_id"] is None:
        return {"result": {"answer": "Please specify which document to summarize.", "tool_used": "none"}}
    summary = summarize_document(state["document_id"])
    return {"result": {"answer": summary, "tool_used": "summarize_tool"}}


def analyze_node(state: AgentState) -> dict:
    if state["document_id"] is None:
        return {"result": {"answer": "Please specify which spreadsheet (document_id) to analyze.", "tool_used": "none"}}
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == state["document_id"]).first()
        filename = doc.filename if doc else None
    finally:
        db.close()
    if not filename or not filename.lower().endswith((".csv", ".xlsx")):
        return {"result": {"answer": "That document isn't a spreadsheet.", "tool_used": "none"}}
    result = analyze_spreadsheet(f"uploads/{filename}", state["query"])
    result["tool_used"] = "analysis_tool"
    return {"result": result}


def pick_route(state: AgentState) -> str:
    return state["route"]


graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("summarize", summarize_node)
graph.add_node("analyze", analyze_node)
graph.set_entry_point("router")
graph.add_conditional_edges(
    "router", pick_route,
    {"retrieval": "retrieval", "summarize": "summarize", "analyze": "analyze"},
)
graph.add_edge("retrieval", END)
graph.add_edge("summarize", END)
graph.add_edge("analyze", END)
agent_app = graph.compile()


def route_query(query: str, document_id: int = None) -> dict:
    final_state = agent_app.invoke(
        {"query": query, "document_id": document_id, "route": "", "result": {}}
    )
    return final_state["result"]