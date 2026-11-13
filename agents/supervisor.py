import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from agents.react import TRANSIENT_MODEL_ERRORS
from config import BACKOFF_PARAMS, MAX_MODEL_ATTEMPTS, MAX_RUNS_PER_SPECIALIST, MAX_SUPERVISOR_STEPS
from state import Contribution, OrchestratorState, SupervisorDecision, request_text

log = logging.getLogger("orquestador")

SUPERVISOR_FAILURES = (OutputParserException, ValidationError, *TRANSIENT_MODEL_ERRORS)

SUPERVISOR_PROMPT = (
    "Eres el supervisor de un equipo de análisis de productos. No buscas datos ni calculas: "
    "decides quién actúa ahora.\n"
    "- researcher: busca productos y sus reseñas en la base de datos.\n"
    "- analyst: calcula estadísticas y sentimiento sobre lo que ya investigó el researcher.\n"
    "- finish: pasa los resultados a validación y redacción final.\n\n"
    "Rúbrica para elegir finish (deben cumplirse las tres):\n"
    "1. La investigación incluye todos los productos que pide el usuario, o confirma que no existen.\n"
    "2. El análisis se hizo sobre la versión vigente de la investigación y cubre esos productos.\n"
    "3. No quedan problemas de validación sin resolver.\n"
    "Si algo falla, delega en el especialista que corresponde con una instrucción concreta y "
    "breve. El analista no puede trabajar sin una investigación. No repitas una tarea que ya "
    "está bien hecha."
)


def describe_research(state: OrchestratorState) -> str:
    research = state.get("research")
    if research is None:
        return "Investigación: todavía no hay."
    products = "; ".join(
        f"{product.nombre} (id {product.producto_id}, {len(product.resenas)} reseñas)"
        for product in research.productos
    )
    return f"Investigación v{research.version}: {products or 'no encontró productos'}."


def describe_analysis(state: OrchestratorState) -> str:
    analysis = state.get("analysis")
    if analysis is None:
        return "Análisis: todavía no hay."
    complete = [item.nombre for item in analysis.productos if item.estadisticas and item.sentimiento]
    return (
        f"Análisis v{analysis.version}, hecho sobre la investigación v{analysis.version_investigacion}; "
        f"productos con estadísticas y sentimiento: {', '.join(complete) or 'ninguno'}."
    )


def describe_validation(state: OrchestratorState) -> str:
    validation = state.get("validation")
    if validation is None or validation.aprobado:
        return "Validación: sin problemas pendientes."
    return "Validación anterior con problemas: " + " ".join(validation.problemas)


def describe_state(state: OrchestratorState) -> str:
    history = "\n".join(
        f"- {item.agente}: {item.detalle}" for item in state.get("contributions", [])
    )
    return "\n".join(
        [
            f"Solicitud del usuario: {request_text(state)}",
            f"Paso del supervisor: {state.get('supervisor_steps', 0) + 1} de {MAX_SUPERVISOR_STEPS}",
            describe_research(state),
            describe_analysis(state),
            describe_validation(state),
            f"Aportes hasta ahora:\n{history or '- ninguno'}",
        ]
    )


def default_decision(state: OrchestratorState) -> SupervisorDecision:
    research = state.get("research")
    analysis = state.get("analysis")
    if research is None:
        return SupervisorDecision(
            siguiente="researcher",
            instruccion=request_text(state),
            motivo="Decisión por defecto: falta la investigación.",
        )
    if research.productos and (analysis is None or analysis.version_investigacion != research.version):
        return SupervisorDecision(
            siguiente="analyst",
            instruccion="Analiza todos los productos investigados.",
            motivo="Decisión por defecto: falta un análisis vigente.",
        )
    return SupervisorDecision(siguiente="finish", instruccion="", motivo="Decisión por defecto: hay datos suficientes.")


def enforce_contracts(
    decision: SupervisorDecision, state: OrchestratorState
) -> tuple[SupervisorDecision, str | None]:
    if decision.siguiente == "analyst" and state.get("research") is None:
        note = "el análisis necesita una investigación previa: se envía al researcher"
        return SupervisorDecision(siguiente="researcher", instruccion=request_text(state), motivo=decision.motivo), note
    runs = state.get("specialist_runs", {})
    if decision.siguiente != "finish" and runs.get(decision.siguiente, 0) >= MAX_RUNS_PER_SPECIALIST:
        note = f"{decision.siguiente} ya trabajó {MAX_RUNS_PER_SPECIALIST} veces: se finaliza"
        return SupervisorDecision(siguiente="finish", instruccion="", motivo=decision.motivo), note
    return decision, None


def make_supervisor_node(model: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    decide = model.with_structured_output(SupervisorDecision, method="json_schema").with_retry(
        retry_if_exception_type=SUPERVISOR_FAILURES,
        wait_exponential_jitter=True,
        exponential_jitter_params=BACKOFF_PARAMS,
        stop_after_attempt=MAX_MODEL_ATTEMPTS,
    )

    async def supervise(state: OrchestratorState) -> dict[str, Any]:
        step = state.get("supervisor_steps", 0) + 1
        note: str | None = None
        if step > MAX_SUPERVISOR_STEPS:
            decision = SupervisorDecision(siguiente="finish", instruccion="", motivo="Se alcanzó el límite de pasos.")
        else:
            try:
                decision = await decide.ainvoke(
                    [SystemMessage(SUPERVISOR_PROMPT), HumanMessage(describe_state(state))]
                )
                if not isinstance(decision, SupervisorDecision):
                    raise OutputParserException("El supervisor respondió sin una decisión estructurada")
            except SUPERVISOR_FAILURES as error:
                log.warning("El supervisor no dio una decisión válida (%s): se usa la regla por defecto", type(error).__name__)
                decision = default_decision(state)
            decision, note = enforce_contracts(decision, state)

        detail = f"-> {decision.siguiente}: {decision.motivo}" + (f" (ajuste: {note})" if note else "")
        return {
            "next_agent": decision.siguiente,
            "instruction": decision.instruccion,
            "supervisor_steps": step,
            "contributions": [Contribution(agente="supervisor", detalle=detail)],
            "messages": [AIMessage(content=detail, name="supervisor")],
        }

    return supervise


def route_after_supervisor(state: OrchestratorState) -> Literal["researcher", "analyst", "validator"]:
    next_agent = state["next_agent"]
    if next_agent == "finish":
        return "validator"
    return next_agent
