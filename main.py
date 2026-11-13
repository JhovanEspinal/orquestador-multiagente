import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from langgraph.errors import GraphRecursionError

from config import TRACES_DIR
from graph import build_graph, run_request
from llm import ConfigError, RoleModels, build_models, model_name, selected_provider
from trace_log import build_trace, format_delegation, write_trace

DEMO_REQUESTS = [
    "Compara lo que opinan los clientes de los audífonos AuraSound X2 y Nimbus Pro, y dime cuál recomiendas.",
    "¿Cómo les va en las reseñas a la cafetera Brava 300 y al molino Molienda Fina?",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "aiosqlite"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def run_requests(
    models: RoleModels, requests: list[str], origin: str, trace_path: Path | None = None
) -> list[dict]:
    app = build_graph(models)
    traces = []
    for request in requests:
        print(f"\n=== Solicitud: {request}")
        state = await run_request(app, request)
        print(f"\nDelegación:\n{format_delegation(state)}")
        print(f"\nRespuesta final:\n{state['final_answer']}")
        traces.append(build_trace(state, origin))
    if trace_path is not None:
        write_trace(trace_path, traces)
        print(f"\nTraza guardada en {trace_path}")
    return traces


def describe_models(provider: str) -> str:
    return f"LLM real: {provider} (supervisor {model_name(provider, 'supervisor')}, especialistas {model_name(provider, 'worker')})"


def main() -> None:
    parser = argparse.ArgumentParser(description="Orquestador multi-agente de análisis de reseñas.")
    parser.add_argument("solicitud", nargs="?", help="pedido del usuario; sin él se ejecutan los de la demo")
    parser.add_argument("--traza", action="store_true", help="guarda la ejecución en traces/")
    args = parser.parse_args()

    load_dotenv()
    configure_logging()
    try:
        models: RoleModels = build_models()
        provider = selected_provider()
        requests = [args.solicitud] if args.solicitud else DEMO_REQUESTS
        trace_path = TRACES_DIR / f"traza_{provider}.json" if args.traza else None
        asyncio.run(run_requests(models, requests, describe_models(provider), trace_path))
    except (ConfigError, ValueError, GraphRecursionError) as exc:
        raise SystemExit(f"No se pudo ejecutar el orquestador: {exc}") from exc


if __name__ == "__main__":
    main()
