from unittest.mock import MagicMock

import agent


def make_state(query, document_id):
    return {"query": query, "document_id": document_id, "route": "", "result": {}}


def test_spreadsheet_documents_always_route_to_analysis(monkeypatch):
    monkeypatch.setattr(agent, "get_document_filename", lambda document_id: "sales.csv")
    state = make_state("How many units of Phone were sold?", 24)
    assert agent.router_node(state) == {"route": "analyze"}


def test_pdf_documents_never_route_to_analysis(monkeypatch):
    monkeypatch.setattr(agent, "get_document_filename", lambda document_id: "handbook.pdf")
    response = MagicMock()
    response.choices[0].message.content = "analyze"
    monkeypatch.setattr(agent.openai_client.chat.completions, "create", lambda **kwargs: response)
    assert agent.router_node(make_state("What is the leave policy?", 21))["route"] == "retrieval"