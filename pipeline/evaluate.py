from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def evaluate(pipeline_output: dict[str, Any], ground_truth_csv: str | Path) -> Metrics:
    predicted = {(str(m["a_id"]), str(m["b_id"])) for m in pipeline_output.get("matches", [])}

    truth_pos: set[tuple[str, str]] = set()
    truth_neg: set[tuple[str, str]] = set()

    with Path(ground_truth_csv).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a_id = str(row["id_a"])
            b_id = str(row["id_b"])
            label = int(row["label"])
            if label == 1:
                truth_pos.add((a_id, b_id))
            else:
                truth_neg.add((a_id, b_id))

    tp = len(predicted & truth_pos)
    fp = len(predicted - truth_pos)
    fn = len(truth_pos - predicted)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positives=fp,
        false_negatives=fn,
    )


def print_metrics(metrics: Metrics, *, title: str = "Evaluation") -> None:
    print(title)
    print("-" * len(title))
    print(f"{'Precision':<14} {metrics.precision:>8.3f}")
    print(f"{'Recall':<14} {metrics.recall:>8.3f}")
    print(f"{'F1':<14} {metrics.f1:>8.3f}")
    print(f"{'False Positives':<14} {metrics.false_positives:>8d}")
    print(f"{'False Negatives':<14} {metrics.false_negatives:>8d}")

