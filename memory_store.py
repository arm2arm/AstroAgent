"""SQLite-backed memory store with optional FTS5 retrieval."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from token_budget import estimate_tokens, truncate_text


@dataclass
class MemoryItem:
    source: str
    path: str | None
    chunk_index: int
    content: str
    metadata: dict


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _safe_query(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_\-\s]", " ", text or "")
    return " ".join(part for part in cleaned.split() if part)


def _chunk_text(text: str, chunk_tokens: int, model: str | None) -> list[str]:
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph, model=model)
        if para_tokens > chunk_tokens:
            truncated = truncate_text(paragraph, chunk_tokens, model=model)
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0
            chunks.append(truncated)
            continue

        if current_tokens + para_tokens > chunk_tokens and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(paragraph)
        current_tokens += para_tokens

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


class MemoryStore:
    def __init__(self, db_path: str, model: str | None = None) -> None:
        self.db_path = db_path
        self.model = model
        _ensure_dir(db_path)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._fts_enabled = False
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY,
                    source TEXT NOT NULL,
                    path TEXT,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT,
                    content_hash TEXT UNIQUE,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_sources (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER,
                    updated_at REAL
                )
                """
            )
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
                    USING fts5(content, source, path, chunk_index, metadata)
                    """
                )
                self._fts_enabled = True
            except sqlite3.OperationalError:
                self._fts_enabled = False
            self.conn.commit()

    @property
    def fts_enabled(self) -> bool:
        return self._fts_enabled

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _record_source(self, path: str, mtime: float, size: int) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO memory_sources(path, mtime, size, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    mtime=excluded.mtime,
                    size=excluded.size,
                    updated_at=excluded.updated_at
                """,
                (path, mtime, size, time.time()),
            )

    def _source_unchanged(self, path: str, mtime: float, size: int) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT mtime, size FROM memory_sources WHERE path = ?", (path,)
            ).fetchone()
        if not row:
            return False
        return float(row["mtime"]) == float(mtime) and int(row["size"]) == int(size)

    def clear_path(self, path: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM memory_items WHERE path = ?", (path,))
            if self._fts_enabled:
                self.conn.execute("DELETE FROM memory_fts WHERE path = ?", (path,))
            self.conn.commit()

    def add_text(
        self,
        *,
        text: str,
        source: str,
        path: Optional[str],
        metadata: Optional[dict],
        chunk_tokens: int,
    ) -> int:
        if not text:
            return 0

        chunks = _chunk_text(text, chunk_tokens, self.model)
        if not chunks:
            return 0

        inserted = 0
        for idx, chunk in enumerate(chunks):
            hasher = hashlib.sha256()
            hasher.update(f"{source}|{path}|{idx}|".encode("utf-8"))
            hasher.update(chunk.encode("utf-8"))
            content_hash = hasher.hexdigest()
            meta_json = json.dumps(metadata or {}, ensure_ascii=True)
            with self._lock:
                cur = self.conn.cursor()
                cur.execute(
                    """
                    INSERT OR IGNORE INTO memory_items
                    (source, path, chunk_index, content, metadata, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source,
                        path,
                        idx,
                        chunk,
                        meta_json,
                        content_hash,
                        time.time(),
                    ),
                )
                if cur.rowcount:
                    inserted += 1
                    if self._fts_enabled:
                        row_id = cur.lastrowid
                        self.conn.execute(
                            """
                            INSERT INTO memory_fts
                            (rowid, content, source, path, chunk_index, metadata)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (row_id, chunk, source, path, idx, meta_json),
                        )
        with self._lock:
            self.conn.commit()
        return inserted

    def index_file(
        self,
        path: str,
        *,
        source: str,
        chunk_tokens: int,
        force: bool = False,
    ) -> int:
        if not os.path.isfile(path):
            return 0

        stat = os.stat(path)
        if not force and self._source_unchanged(path, stat.st_mtime, stat.st_size):
            return 0

        self.clear_path(path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read()
        except Exception:
            return 0

        inserted = self.add_text(
            text=text,
            source=source,
            path=path,
            metadata={"type": "file"},
            chunk_tokens=chunk_tokens,
        )
        self._record_source(path, stat.st_mtime, stat.st_size)
        return inserted

    def index_paths(
        self,
        paths: Iterable[str],
        *,
        source: str,
        chunk_tokens: int,
        force: bool = False,
    ) -> int:
        total = 0
        for raw in paths:
            if not raw:
                continue
            path = os.path.abspath(raw)
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for name in files:
                        if not name.lower().endswith((".md", ".yaml", ".yml", ".txt")):
                            continue
                        total += self.index_file(
                            os.path.join(root, name),
                            source=source,
                            chunk_tokens=chunk_tokens,
                            force=force,
                        )
            else:
                total += self.index_file(
                    path,
                    source=source,
                    chunk_tokens=chunk_tokens,
                    force=force,
                )
        return total

    def search(self, query: str, top_k: int) -> list[MemoryItem]:
        if not query:
            return []

        cleaned = _safe_query(query)
        items: list[MemoryItem] = []

        if cleaned and self._fts_enabled:
            try:
                with self._lock:
                    rows = self.conn.execute(
                        """
                        SELECT content, source, path, chunk_index, metadata
                        FROM memory_fts
                        WHERE memory_fts MATCH ?
                        ORDER BY bm25(memory_fts)
                        LIMIT ?
                        """,
                        (cleaned, top_k),
                    ).fetchall()
                for row in rows:
                    items.append(
                        MemoryItem(
                            source=row["source"],
                            path=row["path"],
                            chunk_index=int(row["chunk_index"]),
                            content=row["content"],
                            metadata=json.loads(row["metadata"] or "{}"),
                        )
                    )
                return items
            except sqlite3.OperationalError:
                pass

        pattern = f"%{query.strip()}%"
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT content, source, path, chunk_index, metadata
                FROM memory_items
                WHERE content LIKE ? OR metadata LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (pattern, pattern, top_k),
            ).fetchall()
        for row in rows:
            items.append(
                MemoryItem(
                    source=row["source"],
                    path=row["path"],
                    chunk_index=int(row["chunk_index"]),
                    content=row["content"],
                    metadata=json.loads(row["metadata"] or "{}"),
                )
            )
        return items


_thread_local = threading.local()


def get_memory_store(db_path: str, model: str | None = None) -> MemoryStore:
    store = getattr(_thread_local, "store", None)
    if store is None or store.db_path != db_path:
        store = MemoryStore(db_path=db_path, model=model)
        _thread_local.store = store
    return store
