from __future__ import annotations
from typing import Optional, Any
import os

try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False


class DatabaseManager:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv('DATABASE_URL') or 'sqlite:///./chroma.sqlite3'
        self.engine: Optional[Engine] = None
        if SQLALCHEMY_AVAILABLE:
            try:
                self.engine = create_engine(self.database_url, connect_args={"check_same_thread": False} if self.database_url.startswith('sqlite') else {})
            except Exception:
                self.engine = None

    def execute(self, sql: str, params: Optional[dict] = None) -> Any:
        if not self.engine:
            raise RuntimeError('No DB engine available')
        with self.engine.connect() as conn:
            res = conn.execute(text(sql), params or {})
            try:
                return res.fetchall()
            except Exception:
                return res.rowcount

    def scalar(self, sql: str, params: Optional[dict] = None) -> Any:
        if not self.engine:
            raise RuntimeError('No DB engine available')
        with self.engine.connect() as conn:
            res = conn.execute(text(sql), params or {})
            return res.scalar()

    def create_table_if_not_exists(self, ddl: str):
        if not self.engine:
            return False
        with self.engine.connect() as conn:
            conn.execute(text(ddl))
            return True


# singleton
db = DatabaseManager()
