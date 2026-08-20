"""Persistent voiceprint database: SQLite + numpy embeddings.

Speakers are matched by cosine similarity against stored mean embeddings.
Above the configured threshold the known name is assigned; otherwise the
speaker stays as SPEAKER_XX until labeled via `meetrec label`.
"""

import sqlite3
import time
from pathlib import Path

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class VoiceprintDB:
    def __init__(self, db_path: Path):
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS speakers (
                   id INTEGER PRIMARY KEY,
                   name TEXT UNIQUE NOT NULL,
                   embedding BLOB NOT NULL,
                   dim INTEGER NOT NULL,
                   num_samples INTEGER NOT NULL DEFAULT 1,
                   created_at REAL NOT NULL,
                   updated_at REAL NOT NULL
               )"""
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def add(self, name: str, embedding: np.ndarray) -> None:
        """Insert a new speaker, or fold the embedding into an existing one."""
        emb = np.asarray(embedding, dtype=np.float32)
        now = time.time()
        row = self._conn.execute(
            "SELECT embedding, dim, num_samples FROM speakers WHERE name = ?",
            (name,)).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO speakers (name, embedding, dim, num_samples,"
                " created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                (name, emb.tobytes(), emb.size, now, now))
        else:
            old = np.frombuffer(row[0], dtype=np.float32)
            n = row[2]
            # running mean keeps the voiceprint stable across meetings
            merged = ((old * n + emb) / (n + 1)).astype(np.float32)
            self._conn.execute(
                "UPDATE speakers SET embedding = ?, num_samples = ?,"
                " updated_at = ? WHERE name = ?",
                (merged.tobytes(), n + 1, now, name))
        self._conn.commit()

    def match(self, embedding: np.ndarray, threshold: float) -> tuple[str, float] | None:
        """Best (name, similarity) above threshold, else None."""
        emb = np.asarray(embedding, dtype=np.float32)
        best: tuple[str, float] | None = None
        for name, blob in self._conn.execute(
                "SELECT name, embedding FROM speakers"):
            score = cosine_similarity(emb, np.frombuffer(blob, dtype=np.float32))
            if score >= threshold and (best is None or score > best[1]):
                best = (name, score)
        return best

    def list_speakers(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, name, dim, num_samples, created_at, updated_at"
            " FROM speakers ORDER BY name").fetchall()
        return [
            {"id": r[0], "name": r[1], "dim": r[2], "num_samples": r[3],
             "created_at": r[4], "updated_at": r[5]}
            for r in rows
        ]

    def rename(self, old: str, new: str) -> bool:
        cur = self._conn.execute(
            "UPDATE speakers SET name = ?, updated_at = ? WHERE name = ?",
            (new, time.time(), old))
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0
