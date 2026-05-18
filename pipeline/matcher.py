from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()


DEFAULT_OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OLLAMA_CHAT_COMPLETIONS_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_OLLAMA_NATIVE_CHAT_URL = "http://localhost:11434/api/chat"


def _env(name: str, default: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


LLM_PROVIDER = _env("LLM_PROVIDER", "openai").lower()
CHAT_COMPLETIONS_URL = _env(
    "LLM_CHAT_COMPLETIONS_URL",
    DEFAULT_OLLAMA_CHAT_COMPLETIONS_URL if LLM_PROVIDER == "ollama" else DEFAULT_OPENAI_CHAT_COMPLETIONS_URL,
)
MODEL = _env("LLM_MODEL", "gpt-4.1")


PROMPT_TEMPLATE = """You are an entity matching expert. Determine if these two records refer to the same real-world entity.

Record A:
{record_a_as_json}

Record B:
{record_b_as_json}

Respond with ONLY a JSON object:
{{
  "match": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one sentence explanation"
}}
"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class MatchResult:
    a_id: str
    b_id: str
    match: bool
    confidence: float
    reason: str


def _approx_tokens(text: str) -> int:
    # Rough rule of thumb: ~4 chars/token for English-ish text.
    return max(1, int(len(text) / 4))


def estimate_pair_cost_usd(record_a: dict[str, Any], record_b: dict[str, Any]) -> float:
    prompt = PROMPT_TEMPLATE.format(
        record_a_as_json=json.dumps(record_a, ensure_ascii=False),
        record_b_as_json=json.dumps(record_b, ensure_ascii=False),
    )
    # Include a small allowance for the JSON response.
    tokens = _approx_tokens(prompt) + 80
    return (tokens / 1000.0) * 0.00015


def _parse_json_only(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        m = _JSON_RE.search(text)
        if not m:
            raise
        return json.loads(m.group(0))


def _ollama_native_chat_url(chat_completions_url: str) -> str:
    """
    Accepts either an Ollama OpenAI-compat URL (/v1/chat/completions) or a native one (/api/chat)
    and returns the native /api/chat URL.
    """
    url = chat_completions_url.strip().rstrip("/")
    if url.endswith("/api/chat"):
        return url
    if url.endswith("/v1/chat/completions"):
        return url[: -len("/v1/chat/completions")] + "/api/chat"
    if "/v1/" in url:
        return url.split("/v1/")[0] + "/api/chat"
    return DEFAULT_OLLAMA_NATIVE_CHAT_URL


def _ollama_base_url(chat_completions_url: str) -> str:
    url = chat_completions_url.strip().rstrip("/")
    if "/api/" in url:
        return url.split("/api/")[0]
    if "/v1/" in url:
        return url.split("/v1/")[0]
    return "http://localhost:11434"


async def _maybe_print_ollama_diagnostics(client: httpx.AsyncClient) -> None:
    if os.getenv("LLM_DIAGNOSTICS", "").strip() not in ("1", "true", "yes", "on"):
        return
    base = _ollama_base_url(CHAT_COMPLETIONS_URL)
    try:
        ver = await client.get(f"{base}/api/version", timeout=5)
        if ver.is_success:
            print(f"[ollama] version: {ver.json().get('version')}")
    except Exception:
        pass
    try:
        ps = await client.get(f"{base}/api/ps", timeout=5)
        if ps.is_success:
            data = ps.json()
            # Schema varies by Ollama version; print a compact hint that helps confirm GPU offload.
            models = data.get("models") if isinstance(data, dict) else None
            if isinstance(models, list) and models:
                m0 = models[0] if isinstance(models[0], dict) else {}
                hint_keys = [k for k in ("model", "name", "size_vram", "gpu", "details") if k in m0]
                hint = {k: m0.get(k) for k in hint_keys}
                print(f"[ollama] ps: {hint}")
            else:
                print("[ollama] ps: (no running models yet)")
    except Exception:
        pass


async def _call_openai_one(
    client: httpx.AsyncClient,
    *,
    api_key: str | None,
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    max_retries: int = 8,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        record_a_as_json=json.dumps(record_a, ensure_ascii=False),
        record_b_as_json=json.dumps(record_b, ensure_ascii=False),
    )

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    max_tokens = _env_int("LLM_MAX_TOKENS", 200)
    timeout_s = _env_float("LLM_REQUEST_TIMEOUT_S", 120.0)
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(max_retries):
        try:
            resp = await client.post(CHAT_COMPLETIONS_URL, headers=headers, json=payload, timeout=timeout_s)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return _parse_json_only(content)
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as exc:
            # Exponential backoff with jitter.
            if attempt == max_retries - 1:
                raise
            base = 0.8 * (2**attempt)
            sleep_s = min(20.0, base + random.random())
            # Add a bit of extra wait for explicit 429.
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 429:
                sleep_s = max(sleep_s, 5.0)
            await asyncio.sleep(sleep_s)

    raise RuntimeError("unreachable")


async def _call_ollama_one(
    client: httpx.AsyncClient,
    *,
    record_a: dict[str, Any],
    record_b: dict[str, Any],
    max_retries: int = 4,
) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        record_a_as_json=json.dumps(record_a, ensure_ascii=False),
        record_b_as_json=json.dumps(record_b, ensure_ascii=False),
    )
    url = _ollama_native_chat_url(CHAT_COMPLETIONS_URL)
    max_tokens = _env_int("LLM_MAX_TOKENS", 200)
    timeout_s = _env_float("LLM_REQUEST_TIMEOUT_S", 300.0)
    options: dict[str, Any] = {
        "temperature": 0,
        "num_ctx": _env_int("OLLAMA_NUM_CTX", 2048),
    }
    num_gpu = os.getenv("OLLAMA_NUM_GPU", "").strip()
    if num_gpu:
        try:
            options["num_gpu"] = int(num_gpu)
        except Exception:
            pass

    payload = {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": prompt}],
        "options": options,
    }

    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, timeout=timeout_s)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "")
            if isinstance(content, str) and len(content) > max_tokens * 8:
                content = content[: max_tokens * 8]
            return _parse_json_only(str(content))
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as exc:
            if attempt == max_retries - 1:
                raise
            base = 0.6 * (2**attempt)
            sleep_s = min(10.0, base + random.random())
            if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None and exc.response.status_code == 429:
                sleep_s = max(sleep_s, 2.5)
            await asyncio.sleep(sleep_s)

    raise RuntimeError("unreachable")


async def verify_candidates(
    candidate_pairs: list[tuple[str, str, float]],
    *,
    records_a_by_id: dict[str, dict[str, Any]],
    records_b_by_id: dict[str, dict[str, Any]],
    threshold: float = 0.85,
    batch_size: int = 20,
    concurrency: int = 20,
) -> tuple[list[MatchResult], float]:
    """
    Stage 2: LLM verification via OpenAI API.

    Returns (matches, estimated_cost_usd).
    """
    api_key = os.getenv("LLM_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    # Cost estimate is based on prompt sizes; doesn't require API response.
    est_cost_usd = 0.0
    for a_id, b_id, _sim in candidate_pairs:
        est_cost_usd += estimate_pair_cost_usd(records_a_by_id[a_id], records_b_by_id[b_id])

    if LLM_PROVIDER == "openai" and not api_key:
        # Non-interactive fallback: allow the pipeline to run end-to-end without external calls.
        # This is useful for smoke tests and report generation when a key is not configured.
        matches = []
        for a_id, b_id, sim in candidate_pairs:
            conf = float(sim)
            if conf >= threshold:
                matches.append(
                    MatchResult(
                        a_id=a_id,
                        b_id=b_id,
                        match=True,
                        confidence=conf,
                        reason="MOCK verifier (OPENAI_API_KEY not set): using embedding cosine similarity.",
                    )
                )
        return matches, float(est_cost_usd)

    if LLM_PROVIDER == "ollama":
        # Prevent host lockups: local inference + high parallelism can saturate CPU/GPU and trigger paging.
        batch_size = _env_int("LLM_BATCH_SIZE", 2)
        concurrency = _env_int("LLM_CONCURRENCY", 1)
    else:
        batch_size = _env_int("LLM_BATCH_SIZE", batch_size)
        concurrency = _env_int("LLM_CONCURRENCY", concurrency)

    semaphore = asyncio.Semaphore(max(1, concurrency))
    matches: list[MatchResult] = []

    async with httpx.AsyncClient() as client:
        if LLM_PROVIDER == "ollama":
            await _maybe_print_ollama_diagnostics(client)

        async def _run_one(a_id: str, b_id: str) -> MatchResult | None:
            async with semaphore:
                record_a = records_a_by_id[a_id]
                record_b = records_b_by_id[b_id]
                if LLM_PROVIDER == "ollama":
                    obj = await _call_ollama_one(client, record_a=record_a, record_b=record_b)
                else:
                    obj = await _call_openai_one(client, api_key=api_key or None, record_a=record_a, record_b=record_b)
                match = bool(obj.get("match", False))
                conf = float(obj.get("confidence", 0.0))
                reason = str(obj.get("reason", "")).strip()
                if match and conf >= threshold:
                    return MatchResult(a_id=a_id, b_id=b_id, match=True, confidence=conf, reason=reason)
                return None

        # Batch scheduling in groups to avoid huge task bursts.
        for i in range(0, len(candidate_pairs), max(1, batch_size)):
            batch = candidate_pairs[i : i + batch_size]
            tasks = [asyncio.create_task(_run_one(a_id, b_id)) for a_id, b_id, _sim in batch]
            for coro in asyncio.as_completed(tasks):
                res = await coro
                if res is not None:
                    matches.append(res)

    return matches, float(est_cost_usd)
