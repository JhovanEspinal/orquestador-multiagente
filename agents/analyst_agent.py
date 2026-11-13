import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field

from agents.react import (
    SPECIALIST_FAILURES,
    build_react_agent,
    failed_run,
    final_text,
    run_react_agent,
    successful_tool_results,
)
from analytics import rating_statistics, review_sentiment
from state import (
    AnalysisReport,
    Contribution,
    OrchestratorState,
    ProductAnalysis,
    ProductFindings,
    RatingStats,
    ResearchReport,
    SentimentSummary,
)

log = logging.getLogger("orquestador")

ANALYST_PROMPT = (
    "Eres el agente de análisis. Procesas los datos que ya reunió el investigador: para cada "
    "producto de la lista llama a estadisticas_calificaciones y a sentimiento_resenas. No "
    "busques productos nuevos ni inventes números: usa solo lo que devuelven tus herramientas. "
    "Termina con un resumen breve que compare los productos con esos números."
)


class ProductoInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    producto_id: int = Field(gt=0, description="ID de un producto de la lista investigada.")


def build_analyst_tools(research: ResearchReport) -> list[BaseTool]:
    products = {product.producto_id: product for product in research.productos}

    def unknown_product(producto_id: int) -> dict[str, Any]:
        return {
            "estado": "sin_resultados",
            "producto_id": producto_id,
            "productos_disponibles": sorted(products),
        }

    @tool(args_schema=ProductoInput)
    async def estadisticas_calificaciones(producto_id: int) -> dict[str, Any]:
        """Calcula las estadísticas de las calificaciones de un producto investigado: cantidad,
        promedio, mediana, desviación estándar, porcentaje de reseñas con 4 o 5 estrellas y
        distribución por estrellas.

        Úsala una vez por cada producto de la lista. Los datos salen de la investigación vigente,
        así que solo necesitas el producto_id.
        """
        product = products.get(producto_id)
        if product is None:
            return unknown_product(producto_id)
        stats = rating_statistics([review.calificacion for review in product.resenas])
        return {"estado": "encontrado", "producto_id": producto_id, "estadisticas": stats.model_dump()}

    @tool(args_schema=ProductoInput)
    async def sentimiento_resenas(producto_id: int) -> dict[str, Any]:
        """Clasifica el sentimiento de las reseñas de un producto investigado como positivas,
        neutras o negativas, y devuelve el puntaje medio de polaridad entre -1 y 1.

        Usa un léxico de palabras positivas y negativas, no un modelo de lenguaje. Úsala una vez
        por cada producto de la lista; solo necesitas el producto_id.
        """
        product = products.get(producto_id)
        if product is None:
            return unknown_product(producto_id)
        sentiment = review_sentiment([review.texto for review in product.resenas])
        return {"estado": "encontrado", "producto_id": producto_id, "sentimiento": sentiment.model_dump()}

    return [estadisticas_calificaciones, sentimiento_resenas]


def analyst_task(instruction: str, research: ResearchReport) -> str:
    products = "\n".join(
        f"- producto_id {product.producto_id}: {product.nombre} ({len(product.resenas)} reseñas)"
        for product in research.productos
    )
    return (
        f"Instrucción del supervisor: {instruction}\n\n"
        f"Productos investigados (investigación v{research.version}):\n{products}"
    )


def product_analysis(
    product: ProductFindings,
    stats: dict[int, RatingStats],
    sentiment: dict[int, SentimentSummary],
) -> ProductAnalysis:
    return ProductAnalysis(
        producto_id=product.producto_id,
        nombre=product.nombre,
        estadisticas=stats.get(product.producto_id),
        sentimiento=sentiment.get(product.producto_id),
    )


def make_analyst_node(model: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    async def analyze(state: OrchestratorState) -> dict[str, Any]:
        research = state.get("research")
        if research is None:
            return {
                "specialist_runs": {"analyst": 1},
                "contributions": [Contribution(agente="analyst", detalle="Sin investigación que analizar")],
            }
        agent = build_react_agent(model, build_analyst_tools(research), ANALYST_PROMPT)
        task = analyst_task(state.get("instruction") or "Analiza todos los productos.", research)
        try:
            messages = await run_react_agent(agent, task)
        except SPECIALIST_FAILURES as error:
            log.warning("El analista falló: %s", type(error).__name__)
            return failed_run("analyst", error)

        stats = {
            result["producto_id"]: RatingStats.model_validate(result["estadisticas"])
            for result in successful_tool_results(messages, "estadisticas_calificaciones")
        }
        sentiment = {
            result["producto_id"]: SentimentSummary.model_validate(result["sentimiento"])
            for result in successful_tool_results(messages, "sentimiento_resenas")
        }
        previous = state.get("analysis")
        report = AnalysisReport(
            version=(previous.version if previous else 0) + 1,
            version_investigacion=research.version,
            productos=[product_analysis(product, stats, sentiment) for product in research.productos],
            resumen=final_text(messages),
        )
        covered = ", ".join(
            f"{item.nombre} (promedio {item.estadisticas.promedio})"
            for item in report.productos
            if item.estadisticas is not None
        )
        return {
            "analysis": report,
            "specialist_runs": {"analyst": 1},
            "contributions": [
                Contribution(
                    agente="analyst",
                    detalle=f"Análisis v{report.version} sobre investigación v{research.version}: {covered or 'sin estadísticas'}",
                )
            ],
            "messages": [AIMessage(content=report.resumen, name="analyst")],
        }

    return analyze
