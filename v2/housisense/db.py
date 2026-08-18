from sqlalchemy import create_engine, text

from .config import DB_URL

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL)
    return _engine


def run(sql, **params):
    with engine().begin() as con:
        con.execute(text(sql), params or None)


def enable_extensions():
    run("CREATE EXTENSION IF NOT EXISTS postgis;")
    run("CREATE EXTENSION IF NOT EXISTS vector;")
