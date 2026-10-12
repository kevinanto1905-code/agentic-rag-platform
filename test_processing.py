import json
from unittest.mock import MagicMock

import pytest

import processing


# ---------- chunking ----------

def test_chunk_text_splits_long_text():
    chunks = processing.chunk_text("word " * 1000, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 100 for c in chunks)


def test_chunk_text_overlap_is_shared_between_neighbours():
    words = [f"w{i}" for i in range(300)]
    chunks = processing.chunk_text(" ".join(words), chunk_size=100, overlap=20)
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]


def test_chunk_text_empty_input_returns_no_chunks():
    assert processing.chunk_text("") == []


# ---------- spreadsheet analysis ----------

@pytest.fixture
def sales_csv(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(
        "Region,Product,Units,Revenue\n"
        "North,Laptop,10,500000\n"
        "North,Phone,25,300000\n"
        "South,Phone,30,360000\n"
        "West,Phone,20,240000\n"
    )
    return str(path)


def fake_llm_plan(monkeypatch, plan):
    """Make the mocked LLM return a fixed JSON plan."""
    response = MagicMock()
    response.choices[0].message.content = json.dumps(plan)
    monkeypatch.setattr(
        processing.openai_client.chat.completions, "create", lambda **kwargs: response
    )


def plan(operation, column=None, group_by=None, filter_column=None, filter_value=None):
    return {
        "operation": operation,
        "column": column,
        "group_by": group_by,
        "filter_column": filter_column,
        "filter_value": filter_value,
    }


def test_sum_of_a_column(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("sum", "Revenue"))
    result = processing.analyze_spreadsheet(sales_csv, "total revenue?")
    assert result["result"] == 1400000


def test_sum_grouped_by_region(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("sum", "Revenue", group_by="Region"))
    result = processing.analyze_spreadsheet(sales_csv, "revenue by region?")
    assert result["result"] == {"North": 800000, "South": 360000, "West": 240000}


def test_filter_is_case_insensitive(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("sum", "Units", filter_column="Product", filter_value="phone"))
    result = processing.analyze_spreadsheet(sales_csv, "phone units?")
    assert result["result"] == 75


def test_count_rows_per_group(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("count", group_by="Region"))
    result = processing.analyze_spreadsheet(sales_csv, "rows per region?")
    assert result["result"] == {"North": 2, "South": 1, "West": 1}


def test_unsupported_operation_is_rejected(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("delete", "Revenue"))
    result = processing.analyze_spreadsheet(sales_csv, "delete everything")
    assert "isn't supported" in result["answer"]


def test_unknown_column_is_rejected(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("sum", "Profit"))
    result = processing.analyze_spreadsheet(sales_csv, "total profit?")
    assert "couldn't find a column" in result["answer"]


def test_non_numeric_column_is_rejected(sales_csv, monkeypatch):
    fake_llm_plan(monkeypatch, plan("sum", "Product"))
    result = processing.analyze_spreadsheet(sales_csv, "sum of products?")
    assert "isn't a numeric column" in result["answer"]