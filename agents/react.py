import json
from collections.abc import Sequence
from typing import Any

import anthropic
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from config import (
    MAX_MODEL_ATTEMPTS,
    MAX_SUMMARY_CHARS,
    MIN_STEPS_TO_CALL_TOOLS,
    SPECIALIST_RECURSION_LIMIT,
)
from state import AgentName, Contribution

TRANSIENT_MODEL_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
    ConnectionError,
    TimeoutError,
)
SPECIALIST_FAILURES = (GraphRecursionError, *TRANSIENT_MODEL_ERRORS)
STEP_LIMIT_MESSAGE = "Alcancé el límite de pasos sin terminar la tarea."


class SpecialistState(MessagesState):
    remaining_steps: RemainingSteps


def build_react_agent(
    model: BaseChatModel, tools: Sequence[BaseTool], system_prompt: str
) -> CompiledStateGraph:
    bound_model = model.bind_tools(tools)

    async def call_model(state: SpecialistState) -> dict[str, list[AIMessage]]:
        response = await bound_model.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        if response.tool_calls and state["remaining_steps"] < MIN_STEPS_TO_CALL_TOOLS:
            return {"messages": [AIMessage(id=response.id, content=STEP_LIMIT_MESSAGE)]}
        return {"messages": [response]}

    graph = StateGraph(SpecialistState)
    graph.add_node(
        "agent",
        call_model,
        retry_policy=RetryPolicy(max_attempts=MAX_MODEL_ATTEMPTS, retry_on=TRANSIENT_MODEL_ERRORS),
    )
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


async def run_react_agent(agent: CompiledStateGraph, task: str) -> list[AnyMessage]:
    result = await agent.ainvoke(
        {"messages": [HumanMessage(task)]}, {"recursion_limit": SPECIALIST_RECURSION_LIMIT}
    )
    return result["messages"]


def successful_tool_results(messages: Sequence[AnyMessage], tool_name: str) -> list[dict[str, Any]]:
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == tool_name and message.status == "success":
            try:
                payload = json.loads(message.content)
            except (TypeError, ValueError):
                continue
            if isinstance(payload, dict) and payload.get("estado") == "encontrado":
                results.append(payload)
    return results


def final_text(messages: Sequence[AnyMessage]) -> str:
    text = str(messages[-1].text).strip() if messages else ""
    return text[:MAX_SUMMARY_CHARS]


def failed_run(agent: AgentName, error: BaseException) -> dict[str, Any]:
    return {
        "specialist_runs": {agent: 1},
        "contributions": [Contribution(agente=agent, detalle=f"Falló: {type(error).__name__}")],
    }
