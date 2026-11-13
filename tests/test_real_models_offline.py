import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agents.research_agent import RESEARCH_TOOLS
from llm import ConfigError, build_chat_model, build_models, model_name
from scripted_model import call, tool_calls
from state import SupervisorDecision

CONVERSATION = [
    SystemMessage("Sistema"),
    HumanMessage("Busca Nimbus Pro"),
    tool_calls(call("buscar_producto", "toolu_1", nombre="Nimbus Pro")),
    ToolMessage('{"estado": "encontrado"}', tool_call_id="toolu_1", name="buscar_producto"),
]


@pytest.fixture
def provider(monkeypatch, request):
    monkeypatch.setenv("LLM_PROVIDER", request.param)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "clave-falsa")
    monkeypatch.setenv("OPENAI_API_KEY", "clave-falsa")
    return request.param


@pytest.mark.parametrize("provider", ["anthropic", "openai"], indirect=True)
def test_worker_request_carries_the_tools_and_the_tool_cycle(provider):
    bound = build_models().researcher.bind_tools(RESEARCH_TOOLS)
    payload = bound.bound._get_request_payload(CONVERSATION, **bound.kwargs)
    names = {tool.get("name") or tool["function"]["name"] for tool in payload["tools"]}
    assert names == {"buscar_producto", "obtener_resenas"}
    if provider == "anthropic":
        assert "temperature" not in payload


def schema_or_text(value):
    return value.model_json_schema() if hasattr(value, "model_json_schema") else str(value)


@pytest.mark.parametrize("provider", ["anthropic", "openai"], indirect=True)
def test_supervisor_request_asks_for_the_decision_schema(provider, recwarn):
    model = build_chat_model(provider, model_name(provider, "supervisor"))
    structured = model.with_structured_output(SupervisorDecision, method="json_schema")
    binding = structured.first
    payload = binding.bound._get_request_payload([HumanMessage("Estado")], **binding.kwargs)
    serialized = json.dumps(payload, ensure_ascii=False, default=schema_or_text)
    assert "siguiente" in serialized and "researcher" in serialized and "finish" in serialized
    assert "tool_choice" not in payload
    assert not [warning for warning in recwarn if "forced tool calling" in str(warning.message)]


def test_supervisor_and_workers_use_different_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "clave-falsa")
    models = build_models()
    assert (models.supervisor.model_name, models.researcher.model_name) == ("gpt-4o", "gpt-4o-mini")
    assert models.researcher is models.analyst is models.synthesizer


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_models()
