from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click
import pandas as pd
from dotenv import load_dotenv

from pipeline.pipeline import run as run_pipeline
from report.generate import generate_report


load_dotenv()


def _load_csv(path: str) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows = df.to_dict(orient="records")
    for i, r in enumerate(rows):
        if "id" not in r or pd.isna(r["id"]):
            r["id"] = str(i)
        else:
            r["id"] = str(r["id"])
    return rows


@click.command()
@click.option("--file", "file_", type=click.Path(exists=True, dir_okay=False), help="Single CSV for dedup mode.")
@click.option("--file_a", type=click.Path(exists=True, dir_okay=False), help="CSV A for match mode.")
@click.option("--file_b", type=click.Path(exists=True, dir_okay=False), help="CSV B for match mode.")
@click.option("--mode", type=click.Choice(["dedup", "match"]), required=True)
@click.option("--threshold", type=float, default=0.85, show_default=True)
@click.option("--top_k", type=int, default=10, show_default=True)
@click.option("--output", type=click.Path(dir_okay=False), default="audit_report.html", show_default=True)
@click.option("--customer", type=str, default="Your CRM", show_default=True)
def main(
    file_: str | None,
    file_a: str | None,
    file_b: str | None,
    mode: str,
    threshold: float,
    top_k: int,
    output: str,
    customer: str,
) -> None:
    t0 = time.time()

    if mode == "dedup":
        if not file_:
            raise click.UsageError("--file is required for --mode dedup")
        source_a = _load_csv(file_)
        source_b = None
    else:
        # Back-compat with the prompt's example that uses --file as the A-side.
        if not file_a and file_:
            file_a = file_
        if not file_a or not file_b:
            raise click.UsageError("--file_a/--file (A) and --file_b (B) are required for --mode match")
        source_a = _load_csv(file_a)
        source_b = _load_csv(file_b)

    out = run_pipeline(source_a, source_b, threshold=threshold, top_k=top_k)
    report_path = generate_report(out, customer_name=customer, output_path=output)

    elapsed = time.time() - t0
    stats = out.get("stats", {})
    print(f"Matches found: {stats.get('matches_found', 0)}")
    print(f"Estimated cost (USD): {stats.get('estimated_cost_usd', 0.0):.4f}")
    print(f"Time elapsed (s): {elapsed:.2f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
