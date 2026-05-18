from __future__ import annotations

import os
import math
import re
import hashlib
from typing import Any

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "faiss import failed. Install faiss-cpu (or faiss-gpu) and ensure it matches your Python version."
    ) from exc


load_dotenv()


_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_DEVICE = "cpu"
_FALLBACK_DIM = 128
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DEFAULT_EMBEDDINGS_MODE = "auto"  # auto|transformer|fallback


def _serialize_record(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for k, v in record.items():
        if v is None:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if k == "id":
            continue
        s = str(v).strip()
        if not s:
            continue
        parts.append(f"{k}: {s}")
    return " | ".join(parts) if parts else ""


def _embed(model: SentenceTransformer, records: list[dict[str, Any]]) -> np.ndarray:
    texts = [_serialize_record(r) for r in records]
    # Normalize embeddings so inner product == cosine similarity.
    embs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    return embs


def _embed_fallback(records: list[dict[str, Any]], *, dim: int = _FALLBACK_DIM) -> np.ndarray:
    """
    Lightweight, dependency-free embedding fallback.

    Uses hashed bag-of-tokens features with L2 normalization so dot product approximates cosine similarity.
    This is less accurate than a transformer encoder but avoids large model downloads/loads (and can
    prevent Windows paging-file OOMs).
    """
    texts = [_serialize_record(r).lower() for r in records]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        # Tokenize and hash into a fixed-width vector.
        for tok in _TOKEN_RE.findall(t):
            h = hashlib.blake2b(tok.encode("utf-8", errors="ignore"), digest_size=8).digest()
            idx = int.from_bytes(h, "little") % dim
            out[i, idx] += 1.0
    # L2 normalize (avoid large intermediate allocations on low-memory systems).
    for i in range(out.shape[0]):
        n = float(np.sqrt(np.dot(out[i], out[i])))
        if n > 0:
            out[i] /= n
    return out


def block_candidates(
    source_a: list[dict[str, Any]],
    source_b: list[dict[str, Any]] | None = None,
    *,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """
    Stage 1: Embedding-based blocking.

    Returns (id_a, id_b, cosine_similarity) candidate pairs. In dedup mode
    (source_b=None), returns only unique pairs (a_id < b_id by index order).
    """
    if not source_a:
        return []

    embeddings_mode = (os.getenv("ENTITY_RESOLVER_EMBEDDINGS") or _DEFAULT_EMBEDDINGS_MODE).strip().lower()
    if embeddings_mode not in ("auto", "transformer", "fallback"):
        embeddings_mode = _DEFAULT_EMBEDDINGS_MODE

    model: SentenceTransformer | None = None
    if embeddings_mode != "fallback":
        # Default to CPU for stability on small/fragmented GPUs; override with ENTITY_RESOLVER_DEVICE=cuda if desired.
        device = (os.getenv("ENTITY_RESOLVER_DEVICE") or _DEFAULT_DEVICE).strip()
        model_name = (os.getenv("ENTITY_RESOLVER_EMBEDDING_MODEL") or _MODEL_NAME).strip()
        try:
            model = SentenceTransformer(model_name, device=device)
        except OSError:
            # Common on Windows when the paging file is constrained or HF cache is corrupted.
            model = None
        except Exception:
            model = None

    if source_b is None:
        records_b = source_a
        dedup = True
    else:
        records_b = source_b
        dedup = False

    if model is None or embeddings_mode == "fallback":
        embs_a = _embed_fallback(source_a)
        embs_b = _embed_fallback(records_b)
    else:
        embs_a = _embed(model, source_a)
        embs_b = _embed(model, records_b)

    index = faiss.IndexFlatIP(embs_b.shape[1])
    index.add(embs_b)

    # Fetch a bit extra in dedup mode to compensate for self-hit filtering.
    k_search = top_k + 1 if dedup else top_k
    scores, neighbors = index.search(embs_a, k_search)

    out: list[tuple[str, str, float]] = []
    for i, (row_scores, row_nei) in enumerate(zip(scores, neighbors, strict=True)):
        id_a = str(source_a[i].get("id", i))
        for sim, j in zip(row_scores.tolist(), row_nei.tolist(), strict=True):
            if j < 0:
                continue
            if dedup:
                if j == i:
                    continue
                # Only keep one direction.
                if j < i:
                    continue
            id_b = str(records_b[j].get("id", j))
            out.append((id_a, id_b, float(sim)))

    return out
