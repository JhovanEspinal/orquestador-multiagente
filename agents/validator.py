from typing import Any, Literal

from config import MAX_SUPERVISOR_STEPS
from state import AnalysisReport, Contribution, OrchestratorState, ResearchReport, ValidationReport


def find_problems(research: ResearchReport | None, analysis: AnalysisReport | None) -> list[str]:
    if research is None:
        return ["No hay investigación."]
    if not research.productos:
        return []
    if analysis is None:
        return ["Falta el análisis de los productos investigados."]

    problems = []
    if analysis.version_investigacion != research.version:
        problems.append(
            f"El análisis usa la investigación v{analysis.version_investigacion}, "
            f"pero la vigente es la v{research.version}."
        )
    analyzed = {item.producto_id: item for item in analysis.productos}
    for product in research.productos:
        result = analyzed.get(product.producto_id)
        if result is None or result.estadisticas is None:
            problems.append(f"Faltan las estadísticas de {product.nombre}.")
        elif result.estadisticas.n != len(product.resenas):
            problems.append(
                f"Las estadísticas de {product.nombre} usan {result.estadisticas.n} reseñas "
                f"y la investigación tiene {len(product.resenas)}."
            )
        if result is None or result.sentimiento is None:
            problems.append(f"Falta el sentimiento de {product.nombre}.")
    return problems


async def validate_results(state: OrchestratorState) -> dict[str, Any]:
    research = state.get("research")
    analysis = state.get("analysis")
    problems = find_problems(research, analysis)
    report = ValidationReport(
        aprobado=not problems,
        problemas=problems,
        version_investigacion=research.version if research else None,
        version_analisis=analysis.version if analysis else None,
    )
    detail = "Aprobado" if report.aprobado else "Rechazado: " + " ".join(problems)
    return {"validation": report, "contributions": [Contribution(agente="validator", detalle=detail)]}


def route_after_validation(state: OrchestratorState) -> Literal["supervisor", "synthesizer"]:
    if state["validation"].aprobado or state.get("supervisor_steps", 0) >= MAX_SUPERVISOR_STEPS:
        return "synthesizer"
    return "supervisor"
