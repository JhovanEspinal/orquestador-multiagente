import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from config import CATALOG_DB_PATH, MAX_PRODUCT_MATCHES, SEED_PATH
from state import ProductFindings, Review
from text_utils import normalize_text

PRODUCTS_BY_NAME_QUERY = """
    SELECT id, nombre, categoria, precio FROM productos
    WHERE normalizar(nombre) LIKE ? ORDER BY id LIMIT ?
"""
PRODUCT_BY_ID_QUERY = "SELECT id, nombre, categoria, precio FROM productos WHERE id = ?"
REVIEWS_QUERY = """
    SELECT id, calificacion, texto, fecha FROM resenas WHERE producto_id = ? ORDER BY fecha
"""

ProductRow = tuple[int, str, str, int]


def ensure_database(path: Path = CATALOG_DB_PATH, seed_path: Path = SEED_PATH) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.executescript(seed_path.read_text(encoding="utf-8"))
    connection.close()


@asynccontextmanager
async def read_only_connection(path: Path = CATALOG_DB_PATH) -> AsyncIterator[aiosqlite.Connection]:
    ensure_database(path)
    async with aiosqlite.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as connection:
        await connection.create_function("normalizar", 1, normalize_text)
        yield connection


async def find_products(name: str, path: Path = CATALOG_DB_PATH) -> list[ProductRow]:
    pattern = f"%{normalize_text(name)}%"
    async with read_only_connection(path) as connection:
        async with connection.execute(PRODUCTS_BY_NAME_QUERY, (pattern, MAX_PRODUCT_MATCHES)) as cursor:
            return list(await cursor.fetchall())


async def get_findings(product_id: int, path: Path = CATALOG_DB_PATH) -> ProductFindings | None:
    async with read_only_connection(path) as connection:
        async with connection.execute(PRODUCT_BY_ID_QUERY, (product_id,)) as cursor:
            product = await cursor.fetchone()
        async with connection.execute(REVIEWS_QUERY, (product_id,)) as cursor:
            reviews = await cursor.fetchall()
    if product is None:
        return None
    return ProductFindings(
        producto_id=product[0],
        nombre=product[1],
        categoria=product[2],
        precio=product[3],
        resenas=[Review(id=row[0], calificacion=row[1], texto=row[2], fecha=row[3]) for row in reviews],
    )
