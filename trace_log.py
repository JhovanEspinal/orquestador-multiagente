import json
from pathlib import Path
from typing import Any

from state import OrchestratorState, request_text


def build_trace(state: OrchestratorState, origin: str) -> dict[str, Any]:
    reports = {
        key: state[key].model_dump() if state.get(key) is not None else None
        for key in ("research", "analysis", "validation")
    }
    return {
        "origen": origin,
        "solicitud": request_text(state),
        "pasos_del_supervisor": state.get("supervisor_steps", 0),
        "delegacion": [item.model_dump() for item in state.get("contributions", [])],
        "investigacion": reports["research"],
        "analisis": reports["analysis"],
        "validacion": reports["validation"],
        "respuesta_final": state.get("final_answer"),
    }


def write_trace(path: Path, traces: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_delegation(state: OrchestratorState) -> str:
    return "\n".join(
        f"{position:>2}. [{item.agente}] {item.detalle}"
        for position, item in enumerate(state.get("contributions", []), start=1)
    )
