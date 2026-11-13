import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.react import TRANSIENT_MODEL_ERRORS
from config import BACKOFF_PARAMS, MAX_MODEL_ATTEMPTS
from state import Contribution, OrchestratorState, request_text

log = logging.getLogger("orquestador")

SYNTHESIS_PROMPT = (
    "Eres el redactor final. Responde la solicitud del usuario en español, en uno o dos "
    "párrafos, usando solo los DATOS que recibes: no agregues productos, cifras ni opiniones "
    "que no estén ahí. Cita los promedios y porcentajes que uses. Si la validación no fue "
    "aprobada, explica qué quedó incompleto. Aclara que las conclusiones salen de las reseñas "
    "de la base de datos."
)


def evidence(state: OrchestratorState) -> dict[str, Any]:
    research = state.get("research")
    analysis = state.get("analysis")
    validation = state.get("validation")
    analyzed = {item.producto_id: item for item in analysis.productos} if analysis else {}
    products = []
    for product in research.productos if research else []:
        result = analyzed.get(product.producto_id)
        products.append(
            {
                "nombre": product.nombre,
                "categoria": product.categoria,
                "precio": product.precio,
                "estadisticas": result.estadisticas.model_dump() if result and result.estadisticas else None,
                "sentimiento": result.sentimiento.model_dump() if result and result.sentimiento else None,
            }
        )
    return {
        "productos": products,
        "validacion_aprobada": bool(validation and validation.aprobado),
        "problemas": validation.problemas if validation else [],
    }


def fallback_answer(state: OrchestratorState) -> str:
    data = evidence(state)
    if not data["productos"]:
        return "No se encontraron productos que coincidan con la solicitud en la base de datos."
    lines = ["Resumen generado sin el modelo de lenguaje, a partir de los datos calculados:"]
    for product in data["productos"]:
        stats = product["estadisticas"]
        if stats:
            lines.append(
                f"- {product['nombre']}: promedio {stats['promedio']} sobre {stats['n']} reseñas, "
                f"{stats['porcentaje_positivas']}% con 4 o 5 estrellas."
            )
        else:
            lines.append(f"- {product['nombre']}: sin estadísticas.")
    if data["problemas"]:
        lines.append("Quedó incompleto: " + " ".join(data["problemas"]))
    return "\n".join(lines)


def make_synthesizer_node(model: BaseChatModel) -> Callable[[OrchestratorState], Awaitable[dict[str, Any]]]:
    writer = model.with_retry(
        retry_if_exception_type=TRANSIENT_MODEL_ERRORS,
        wait_exponential_jitter=True,
        exponential_jitter_params=BACKOFF_PARAMS,
        stop_after_attempt=MAX_MODEL_ATTEMPTS,
    )

    async def synthesize(state: OrchestratorState) -> dict[str, Any]:
        data = json.dumps(evidence(state), ensure_ascii=False)
        try:
            response = await writer.ainvoke(
                [
                    SystemMessage(SYNTHESIS_PROMPT),
                    HumanMessage(f"Solicitud: {request_text(state)}\n\nDATOS:\n{data}"),
                ]
            )
            answer, detail = str(response.text).strip(), "Respuesta final redactada"
        except TRANSIENT_MODEL_ERRORS as error:
            log.warning("El redactor falló (%s): se usa el resumen por defecto", type(error).__name__)
            answer, detail = fallback_answer(state), "Respuesta por defecto: el modelo no respondió"
        return {
            "final_answer": answer,
            "contributions": [Contribution(agente="synthesizer", detalle=detail)],
            "messages": [AIMessage(content=answer, name="synthesizer")],
        }

    return synthesize
