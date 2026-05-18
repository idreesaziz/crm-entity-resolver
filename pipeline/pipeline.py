from __future__ import annotations

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from pipeline.blocker import block_candidates
from pipeline.matcher import verify_candidates


load_dotenv()


def run(
    source_a: list[dict[str, Any]],
    source_b: list[dict[str, Any]] | None,
    threshold: float = 0.85,
    top_k: int = 10,
) -> dict[str, Any]:
    if not source_a:
        return {
            "matches": [],
            "stats": {
                "total_records_a": 0,
                "total_records_b": 0 if source_b is None else len(source_b),
                "candidate_pairs_after_blocking": 0,
                "matches_found": 0,
                "estimated_cost_usd": 0.0,
            },
        }

    # Normalize ids to strings.
    for i, r in enumerate(source_a):
        r["id"] = str(r.get("id", i))
    if source_b is not None:
        for i, r in enumerate(source_b):
            r["id"] = str(r.get("id", i))

    candidates = block_candidates(source_a, source_b, top_k=top_k)
    max_candidates_raw = os.getenv("ENTITY_RESOLVER_MAX_CANDIDATES", "").strip()
    if max_candidates_raw:
        try:
            max_candidates = int(max_candidates_raw)
        except Exception:
            max_candidates = 0
        if max_candidates > 0 and len(candidates) > max_candidates:
            # Keep the strongest candidate pairs by cosine similarity so evaluation remains meaningful.
            candidates = sorted(candidates, key=lambda t: float(t[2]), reverse=True)[:max_candidates]

    records_a_by_id = {str(r["id"]): r for r in source_a}
    if source_b is None:
        records_b_by_id = records_a_by_id
        total_b = len(source_a)
    else:
        records_b_by_id = {str(r["id"]): r for r in source_b}
        total_b = len(source_b)

    # Verify via LLM (async) and attach full records.
    llm_matches, estimated_cost_usd = asyncio.run(
        verify_candidates(
            candidates,
            records_a_by_id=records_a_by_id,
            records_b_by_id=records_b_by_id,
            threshold=threshold,
        )
    )

    matches = [
        {
            "a_id": m.a_id,
            "b_id": m.b_id,
            "confidence": float(m.confidence),
            "reason": m.reason,
            "record_a": records_a_by_id[m.a_id],
            "record_b": records_b_by_id[m.b_id],
        }
        for m in llm_matches
    ]

    return {
        "matches": matches,
        "stats": {
            "total_records_a": len(source_a),
            "total_records_b": total_b,
            "candidate_pairs_after_blocking": len(candidates),
            "matches_found": len(matches),
            "estimated_cost_usd": float(estimated_cost_usd),
        },
    }
