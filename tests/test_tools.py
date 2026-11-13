import sqlite3

import pytest

import database
from agents.analyst_agent import build_analyst_tools
from agents.research_agent import buscar_producto, obtener_resenas
from analytics import polarity, rating_statistics, review_sentiment
from state import ProductFindings, ResearchReport, Review, add_runs, merge_findings


async def test_search_ignores_accents_and_case():
    result = await buscar_producto.ainvoke({"nombre": "nimbus pro"})
    assert result["estado"] == "encontrado"
    assert result["productos"][0]["producto_id"] == 2


async def test_partial_name_can_be_ambiguous():
    result = await buscar_producto.ainvoke({"nombre": "AuraSound"})
    assert result["estado"] == "ambiguo"
    assert {product["producto_id"] for product in result["productos"]} == {1, 5}


async def test_reviews_come_with_the_product_sheet():
    result = await obtener_resenas.ainvoke({"producto_id": 1})
    assert result["producto"]["nombre"] == "AuraSound X2"
    assert len(result["producto"]["resenas"]) == 6


async def test_unknown_product():
    assert (await obtener_resenas.ainvoke({"producto_id": 99}))["estado"] == "sin_resultados"


async def test_invalid_id_is_rejected_before_the_database():
    with pytest.raises(Exception, match="producto_id"):
        await obtener_resenas.ainvoke({"producto_id": 0})


async def test_database_failure_reaches_the_model_as_text(monkeypatch):
    async def fail(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(database, "find_products", fail)
    assert "no respondió" in await buscar_producto.ainvoke({"nombre": "Brava"})


async def test_connection_cannot_write():
    async with database.read_only_connection() as connection:
        with pytest.raises(sqlite3.OperationalError):
            await connection.execute("DELETE FROM resenas")


def test_rating_statistics():
    stats = rating_statistics([5, 4, 5, 4, 2, 5])
    assert (stats.n, stats.promedio, stats.mediana, stats.porcentaje_positivas) == (6, 4.17, 4.5, 83.33)
    assert stats.distribucion == {1: 0, 2: 1, 3: 0, 4: 2, 5: 3}


def test_rating_statistics_without_reviews():
    stats = rating_statistics([])
    assert stats.n == 0 and stats.promedio is None


def test_sentiment_uses_the_lexicon_without_accents():
    assert polarity("Excelente, muy cómodos") == 1
    assert polarity("Llegó roto y nunca respondieron") == -1
    assert polarity("Correctos, nada especial") == 0
    summary = review_sentiment(["Excelente", "Malo", "Normal"])
    assert (summary.positivas, summary.negativas, summary.neutras, summary.puntaje_medio) == (1, 1, 1, 0.0)


def finding(product_id: int, ratings: list[int]) -> ProductFindings:
    return ProductFindings(
        producto_id=product_id,
        nombre=f"Producto {product_id}",
        categoria="prueba",
        precio=1000,
        resenas=[Review(id=index, calificacion=rating, texto="Bueno", fecha="2026-01-01") for index, rating in enumerate(ratings)],
    )


def test_research_refinement_adds_products_and_bumps_the_version():
    first = merge_findings(None, [finding(1, [5])], "uno")
    second = merge_findings(first, [finding(2, [3]), finding(1, [5, 4])], "dos")
    assert (first.version, second.version) == (1, 2)
    assert [product.producto_id for product in second.productos] == [1, 2]
    assert len(second.productos[0].resenas) == 2


def test_specialist_run_counts_are_added():
    assert add_runs({"researcher": 1}, {"researcher": 1, "analyst": 1}) == {"researcher": 2, "analyst": 1}


async def test_analyst_tools_read_only_the_current_research():
    research = ResearchReport(version=1, productos=[finding(7, [5, 3])], resumen="")
    stats_tool, sentiment_tool = build_analyst_tools(research)
    stats = await stats_tool.ainvoke({"producto_id": 7})
    assert stats["estadisticas"]["promedio"] == 4.0
    missing = await sentiment_tool.ainvoke({"producto_id": 8})
    assert missing == {"estado": "sin_resultados", "producto_id": 8, "productos_disponibles": [7]}
