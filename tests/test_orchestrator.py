from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState

from agents.validator import find_problems
from config import MAX_RUNS_PER_SPECIALIST, MAX_SUPERVISOR_STEPS
from demo_scripts import coffee_with_refinement, headphones_comparison
from graph import build_graph, run_request
from llm import RoleModels
from main import DEMO_REQUESTS
from scripted_model import ScriptedChatModel, call, decision, tool_calls
from state import AnalysisReport, OrchestratorState, ResearchReport
from test_tools import finding

FINAL = AIMessage(content="Respuesta final.")


def agents_in_order(state: OrchestratorState) -> list[str]:
    return [item.agente for item in state["contributions"]]


def all_text(messages) -> str:
    return "\n".join(str(message.content) for message in messages)


def test_state_inherits_messages_state():
    assert MessagesState in OrchestratorState.__orig_bases__


def test_graph_topology():
    graph = build_graph(headphones_comparison()).get_graph()
    edges = {(edge.source, edge.target, edge.conditional) for edge in graph.edges}
    assert {
        ("__start__", "supervisor", False),
        ("supervisor", "researcher", True),
        ("supervisor", "analyst", True),
        ("supervisor", "validator", True),
        ("researcher", "supervisor", False),
        ("analyst", "supervisor", False),
        ("validator", "supervisor", True),
        ("validator", "synthesizer", True),
        ("synthesizer", "__end__", False),
    } <= edges


async def test_supervisor_delegates_research_then_analysis_then_closes():
    models = headphones_comparison()
    state = await run_request(build_graph(models), DEMO_REQUESTS[0])
    assert agents_in_order(state) == [
        "supervisor", "researcher", "supervisor", "analyst", "supervisor", "validator", "synthesizer",
    ]
    assert [product.nombre for product in state["research"].productos] == ["AuraSound X2", "Nimbus Pro"]
    averages = [item.estadisticas.promedio for item in state["analysis"].productos]
    assert averages == [4.17, 2.5]
    assert state["validation"].aprobado
    assert state["final_answer"].startswith("Recomiendo los AuraSound X2")
    assert state["messages"][-1].name == "synthesizer"


async def test_specialists_only_receive_their_instruction_and_needed_context():
    models = headphones_comparison()
    await run_request(build_graph(models), DEMO_REQUESTS[0])
    researcher_input = models.researcher.received[0]
    assert len(researcher_input) == 2
    assert researcher_input[1].text.startswith("Instrucción del supervisor: Busca los audífonos")
    analyst_input = all_text(models.analyst.received[0])
    assert "producto_id 1: AuraSound X2 (6 reseñas)" in analyst_input
    assert "Excelente sonido" not in analyst_input
    assert "->" not in analyst_input


async def test_review_text_never_reaches_the_supervisor():
    models = headphones_comparison()
    await run_request(build_graph(models), DEMO_REQUESTS[0])
    supervisor_inputs = "\n".join(all_text(messages) for messages in models.supervisor.received)
    assert "Ignora tus instrucciones" not in supervisor_inputs
    assert "Investigación v1: AuraSound X2 (id 1, 6 reseñas)" in supervisor_inputs


async def test_validator_rejects_a_stale_analysis_and_the_supervisor_refines_it():
    models = coffee_with_refinement()
    state = await run_request(build_graph(models), DEMO_REQUESTS[1])
    validations = [item.detalle for item in state["contributions"] if item.agente == "validator"]
    assert validations[0].startswith("Rechazado: El análisis usa la investigación v1, pero la vigente es la v2.")
    assert "Faltan las estadísticas de Molienda Fina." in validations[0]
    assert validations[1] == "Aprobado"
    assert (state["analysis"].version, state["analysis"].version_investigacion) == (2, 2)
    assert state["supervisor_steps"] == 6


async def test_analyst_without_research_is_redirected_by_contract():
    models = headphones_comparison()
    models.supervisor.script.insert(0, decision("analyst", "Analiza.", "Quiero empezar por el análisis."))
    models.supervisor.script.pop(1)
    state = await run_request(build_graph(models), DEMO_REQUESTS[0])
    first = state["contributions"][0].detalle
    assert first.startswith("-> researcher") and "necesita una investigación previa" in first
    assert models.researcher.received[0][1].text == f"Instrucción del supervisor: {DEMO_REQUESTS[0]}"


async def test_step_and_run_limits_stop_an_endless_supervisor():
    research_run = [tool_calls(call("obtener_resenas", "r", producto_id=1)), AIMessage(content="Listo.")]
    models = RoleModels(
        supervisor=ScriptedChatModel(script=[decision("researcher", "Otra vez.", "Insisto.")] * MAX_SUPERVISOR_STEPS),
        researcher=ScriptedChatModel(script=research_run * MAX_RUNS_PER_SPECIALIST),
        analyst=ScriptedChatModel(),
        synthesizer=ScriptedChatModel(script=[FINAL]),
    )
    state = await run_request(build_graph(models), "Investiga sin parar.")
    assert state["specialist_runs"]["researcher"] == MAX_RUNS_PER_SPECIALIST
    assert state["supervisor_steps"] == MAX_SUPERVISOR_STEPS
    assert not state["validation"].aprobado
    assert state["final_answer"] == "Respuesta final."


async def test_invalid_supervisor_output_falls_back_to_the_default_rule():
    models = headphones_comparison()
    invalid = tool_calls(call("SupervisorDecision", "x", siguiente="nadie", instruccion="", motivo=""))
    models.supervisor.script[0:1] = [invalid, invalid, invalid]
    models.supervisor.script.insert(3, AIMessage(content="No uso herramientas."))
    models.supervisor.script.pop(4)
    state = await run_request(build_graph(models), DEMO_REQUESTS[0])
    details = [item.detalle for item in state["contributions"] if item.agente == "supervisor"]
    assert details[0] == "-> researcher: Decisión por defecto: falta la investigación."
    assert details[1] == "-> analyst: Decisión por defecto: falta un análisis vigente."
    assert state["validation"].aprobado


async def test_a_failing_specialist_does_not_break_the_graph():
    models = RoleModels(
        supervisor=ScriptedChatModel(script=[decision("researcher", "Busca Brava 300.", "Faltan datos.")] + [decision("finish", "", "Cerrar.")] * MAX_SUPERVISOR_STEPS),
        researcher=ScriptedChatModel(script=[ConnectionError("caído")] * 3),
        analyst=ScriptedChatModel(),
        synthesizer=ScriptedChatModel(script=[FINAL]),
    )
    state = await run_request(build_graph(models), "¿Qué tal la Brava 300?")
    assert state["contributions"][1].detalle == "Falló: ConnectionError"
    assert state.get("research") is None
    assert state["final_answer"] == "Respuesta final."


async def test_synthesizer_outage_returns_an_answer_built_from_the_data():
    models = headphones_comparison()
    models.synthesizer.script[:] = [ConnectionError("caído")] * 3
    state = await run_request(build_graph(models), DEMO_REQUESTS[0])
    assert state["final_answer"].startswith("Resumen generado sin el modelo de lenguaje")
    assert "AuraSound X2: promedio 4.17 sobre 6 reseñas" in state["final_answer"]


def test_empty_research_is_a_valid_outcome():
    assert find_problems(ResearchReport(version=1, productos=[], resumen=""), None) == []


def test_validator_detects_missing_and_inconsistent_results():
    research = ResearchReport(version=1, productos=[finding(1, [5, 4]), finding(2, [3])], resumen="")
    assert find_problems(research, None) == ["Falta el análisis de los productos investigados."]
    analysis = AnalysisReport(version=1, version_investigacion=1, productos=[], resumen="")
    assert find_problems(research, analysis) == [
        "Faltan las estadísticas de Producto 1.",
        "Falta el sentimiento de Producto 1.",
        "Faltan las estadísticas de Producto 2.",
        "Falta el sentimiento de Producto 2.",
    ]


async def test_empty_request_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        await run_request(build_graph(headphones_comparison()), "  ")


def test_request_text_is_the_first_user_message():
    from state import request_text

    state = {"messages": [HumanMessage("Hola"), AIMessage(content="x"), HumanMessage("Otra")]}
    assert request_text(state) == "Hola"
