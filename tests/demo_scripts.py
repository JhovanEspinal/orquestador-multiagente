from langchain_core.messages import AIMessage

from llm import RoleModels
from scripted_model import ScriptedChatModel, call, decision, tool_calls

ORIGIN = (
    "MODELO GUIONIZADO, NO ES UN LLM REAL: las decisiones del supervisor y los textos de los "
    "agentes están escritos a mano en tests/demo_scripts.py. El grafo, las herramientas, la base "
    "SQLite, las estadísticas y el validador sí se ejecutaron de verdad."
)


def headphones_comparison() -> RoleModels:
    return RoleModels(
        supervisor=ScriptedChatModel(
            script=[
                decision("researcher", "Busca los audífonos AuraSound X2 y Nimbus Pro y obtén todas sus reseñas.", "Faltan los datos de ambos productos."),
                decision("analyst", "Calcula estadísticas y sentimiento de los dos productos investigados.", "La investigación ya cubre los dos productos."),
                decision("finish", "", "La investigación y el análisis vigentes cubren ambos productos."),
            ]
        ),
        researcher=ScriptedChatModel(
            script=[
                tool_calls(call("buscar_producto", "r1", nombre="AuraSound X2"), call("buscar_producto", "r2", nombre="Nimbus Pro")),
                tool_calls(call("obtener_resenas", "r3", producto_id=1), call("obtener_resenas", "r4", producto_id=2)),
                AIMessage(content="Encontré AuraSound X2 (6 reseñas) y Nimbus Pro (6 reseñas)."),
            ]
        ),
        analyst=ScriptedChatModel(
            script=[
                tool_calls(
                    call("estadisticas_calificaciones", "a1", producto_id=1),
                    call("estadisticas_calificaciones", "a2", producto_id=2),
                    call("sentimiento_resenas", "a3", producto_id=1),
                    call("sentimiento_resenas", "a4", producto_id=2),
                ),
                AIMessage(content="AuraSound X2 promedia 4,17 con 83,33% de reseñas de 4 o 5 estrellas; Nimbus Pro promedia 2,5 con 16,67%."),
            ]
        ),
        synthesizer=ScriptedChatModel(
            script=[
                AIMessage(
                    content=(
                        "Recomiendo los AuraSound X2. Tienen un promedio de 4,17 estrellas en 6 reseñas, el 83,33% "
                        "son de 4 o 5 estrellas y 5 de 6 reseñas tienen tono positivo. Los Nimbus Pro promedian 2,5 "
                        "en 6 reseñas, solo el 16,67% llega a 4 o 5 estrellas y su sentimiento es mayormente neutro "
                        "o negativo (1 positiva, 3 neutras y 2 negativas). Estas conclusiones salen de las reseñas "
                        "de la base de datos."
                    )
                )
            ]
        ),
    )


def coffee_with_refinement() -> RoleModels:
    return RoleModels(
        supervisor=ScriptedChatModel(
            script=[
                decision("researcher", "Busca la cafetera Brava 300 y obtén sus reseñas.", "Todavía no hay investigación."),
                decision("analyst", "Calcula estadísticas y sentimiento de la Brava 300.", "Ya hay datos de la Brava 300."),
                decision("researcher", "Falta el molino Molienda Fina: búscalo y obtén sus reseñas.", "La solicitud también pide el molino Molienda Fina."),
                decision("finish", "", "Ya están los dos productos."),
                decision("analyst", "Recalcula estadísticas y sentimiento sobre la investigación v2.", "La validación detectó un análisis desactualizado."),
                decision("finish", "", "El análisis ya usa la investigación vigente."),
            ]
        ),
        researcher=ScriptedChatModel(
            script=[
                tool_calls(call("buscar_producto", "r1", nombre="Brava 300")),
                tool_calls(call("obtener_resenas", "r2", producto_id=3)),
                AIMessage(content="Encontré la cafetera Brava 300 con 5 reseñas."),
                tool_calls(call("buscar_producto", "r3", nombre="Molienda Fina")),
                tool_calls(call("obtener_resenas", "r4", producto_id=4)),
                AIMessage(content="Agregué el molino Molienda Fina con 4 reseñas."),
            ]
        ),
        analyst=ScriptedChatModel(
            script=[
                tool_calls(call("estadisticas_calificaciones", "a1", producto_id=3), call("sentimiento_resenas", "a2", producto_id=3)),
                AIMessage(content="Brava 300 promedia 4,4 con 80% de reseñas de 4 o 5 estrellas."),
                tool_calls(
                    call("estadisticas_calificaciones", "a3", producto_id=3),
                    call("estadisticas_calificaciones", "a4", producto_id=4),
                    call("sentimiento_resenas", "a5", producto_id=3),
                    call("sentimiento_resenas", "a6", producto_id=4),
                ),
                AIMessage(content="Brava 300 promedia 4,4 (80% positivas); Molienda Fina promedia 3,25 (50%)."),
            ]
        ),
        synthesizer=ScriptedChatModel(
            script=[
                AIMessage(
                    content=(
                        "La cafetera Brava 300 tiene mejor recepción: promedio de 4,4 estrellas en 5 reseñas, el 80% "
                        "con 4 o 5 estrellas y ninguna reseña de tono negativo. El molino Molienda Fina promedia 3,25 "
                        "en 4 reseñas, con el 50% de 4 o 5 estrellas y un sentimiento dividido (2 positivas y 2 "
                        "negativas). Estas conclusiones salen de las reseñas de la base de datos."
                    )
                )
            ]
        ),
    )
