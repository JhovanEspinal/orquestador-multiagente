from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.analyst_agent import make_analyst_node
from agents.research_agent import make_research_node
from agents.supervisor import make_supervisor_node, route_after_supervisor
from agents.synthesizer import make_synthesizer_node
from agents.validator import route_after_validation, validate_results
from config import GRAPH_RECURSION_LIMIT
from llm import RoleModels
from state import OrchestratorState


def build_graph(models: RoleModels) -> CompiledStateGraph:
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("supervisor", make_supervisor_node(models.supervisor))
    workflow.add_node("researcher", make_research_node(models.researcher))
    workflow.add_node("analyst", make_analyst_node(models.analyst))
    workflow.add_node("validator", validate_results)
    workflow.add_node("synthesizer", make_synthesizer_node(models.synthesizer))

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_after_supervisor)
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_conditional_edges("validator", route_after_validation)
    workflow.add_edge("synthesizer", END)
    return workflow.compile()


async def run_request(app: CompiledStateGraph, request: str) -> OrchestratorState:
    if not request.strip():
        raise ValueError("La solicitud no puede estar vacía")
    return await app.ainvoke(
        {"messages": [HumanMessage(request)]}, {"recursion_limit": GRAPH_RECURSION_LIMIT}
    )
