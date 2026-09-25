"""Policy retrieval over data/policies/*.md.

Small corpus on purpose: retrieval quality is not what this project is proving,
so it stays a few dozen chunks that a reviewer can read end to end and check.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache

from .config import CHROMA_DIR, POLICY_DIR

COLLECTION = "policies"


def _chunks() -> list[tuple[str, str, str]]:
    """(id, text, source) per paragraph. Paragraph-level is enough at this size;
    a sliding window would add cost with nothing to show for it."""
    out: list[tuple[str, str, str]] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        for para in (p.strip() for p in path.read_text().split("\n\n")):
            if len(para) < 40:
                continue
            uid = hashlib.sha1(f"{path.name}:{para}".encode()).hexdigest()[:16]
            out.append((uid, para, path.stem))
    return out


@lru_cache(maxsize=1)
def _collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.get_or_create_collection(COLLECTION)
    rows = _chunks()
    if not rows:
        raise RuntimeError(f"No policy documents found in {POLICY_DIR}")
    if col.count() != len(rows):
        # Rebuild wholesale - the corpus is tiny and partial state is worse
        # than a two-second reindex.
        if col.count():
            client.delete_collection(COLLECTION)
            col = client.get_or_create_collection(COLLECTION)
        ids, docs, sources = zip(*rows)
        col.add(ids=list(ids), documents=list(docs),
                metadatas=[{"source": s} for s in sources])
    return col


def retrieve(query: str, k: int = 3) -> list[dict[str, str]]:
    res = _collection().query(query_texts=[query], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    return [
        {"text": d, "source": (m or {}).get("source", "unknown")}
        for d, m in zip(docs, metas)
    ]
