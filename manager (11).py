# backend/rag/keyword_index.py
from __future__ import annotations

import threading
from typing import Optional

from backend.utils.normalizer import QueryNormalizer


class KeywordIndex:
    def __init__(self):
        self.index: dict[str, set[str]] = {}
        self.reverse_index: dict[str, set[str]] = {}
        self.chunks: dict[str, str] = {}
        self.chunk_sources: dict[str, str] = {}
        self.lock = threading.Lock()

    def add_chunk(self, chunk_id: str, text: str, source: str):
        with self.lock:
            codes = QueryNormalizer.extract_codes(text)
            self.chunks[chunk_id] = text
            self.chunk_sources[chunk_id] = source
            self.reverse_index[chunk_id] = set()

            for code in codes:
                normalized = QueryNormalizer.normalize_code(code)
                if normalized not in self.index:
                    self.index[normalized] = set()
                self.index[normalized].add(chunk_id)
                self.reverse_index[chunk_id].add(normalized)

    def remove_chunk(self, chunk_id: str):
        with self.lock:
            if chunk_id in self.reverse_index:
                for code in self.reverse_index[chunk_id]:
                    if code in self.index:
                        self.index[code].discard(chunk_id)
                        if not self.index[code]:
                            del self.index[code]
                del self.reverse_index[chunk_id]

            self.chunks.pop(chunk_id, None)
            self.chunk_sources.pop(chunk_id, None)

    def search(self, query: str) -> list[tuple[str, float]]:
        codes = QueryNormalizer.extract_codes(query)
        if not codes:
            return []

        with self.lock:
            chunk_scores: dict[str, int] = {}
            for code in codes:
                normalized = QueryNormalizer.normalize_code(code)
                if normalized in self.index:
                    for chunk_id in self.index[normalized]:
                        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + 1

            results = [
                (chunk_id, score / len(codes))
                for chunk_id, score in chunk_scores.items()
            ]
            results.sort(key=lambda x: x[1], reverse=True)
            return results

    def get_chunk(self, chunk_id: str) -> Optional[str]:
        return self.chunks.get(chunk_id)

    def get_source(self, chunk_id: str) -> str:
        return self.chunk_sources.get(chunk_id, "unknown")

    def clear(self):
        with self.lock:
            self.index.clear()
            self.reverse_index.clear()
            self.chunks.clear()
            self.chunk_sources.clear()

    def get_stats(self) -> dict:
        with self.lock:
            return {
                "total_codes": len(self.index),
                "total_chunks": len(self.chunks),
                "sample_codes": list(self.index.keys())[:20]
            }
