from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def call(name: str, call_id: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "args": arguments, "id": call_id, "type": "tool_call"}


def tool_calls(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


def decision(siguiente: str, instruccion: str, motivo: str) -> AIMessage:
    return tool_calls(
        call("SupervisorDecision", f"decision-{siguiente}", siguiente=siguiente, instruccion=instruccion, motivo=motivo)
    )


class ScriptedChatModel(BaseChatModel):
    script: list[Any] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def with_structured_output(self, schema: Any, *, method: str | None = None, **kwargs: Any) -> Any:
        return super().with_structured_output(schema, **kwargs)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.received.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return ChatResult(generations=[ChatGeneration(message=step)])
