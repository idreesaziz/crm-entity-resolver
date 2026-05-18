from __future__ import annotations

import csv
import io
import json
import sys
import shutil
import zipfile
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm


load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Ditto S3 mirrors previously used by many ER tutorials have become unstable (404 as of 2026-05-17).
# Use the DeepMatcher hosting (University of Wisconsin) and extract the ER-Magellan formatted folders.
#
# Source index: https://raw.githubusercontent.com/anhaidgroup/deepmatcher/master/Datasets.md
DATASETS: dict[str, tuple[str, str]] = {
    # Comes from Dirty.zip -> Dirty/DBLP-ACM/...
    "dblp_acm": ("https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Dirty.zip", "Dirty/DBLP-ACM/"),
    # Comes from Structured.zip -> Structured/Amazon-Google/...
    "amazon_google": ("https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured.zip", "Structured/Amazon-Google/"),
}


def _download_bytes(url: str) -> bytes:
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        with client.stream("GET", url) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", "0") or "0")
            buf = io.BytesIO()
            bar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}")
            for chunk in r.iter_bytes():
                buf.write(chunk)
                bar.update(len(chunk))
            bar.close()
            return buf.getvalue()


def _read_csv_from_zip(z: zipfile.ZipFile, member: str) -> list[dict[str, Any]]:
    with z.open(member) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", newline="")
        reader = csv.DictReader(text)
        return [dict(row) for row in reader]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _standardize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        rr = dict(r)
        rr["id"] = str(rr.get("id") or rr.get("ID") or rr.get("Id") or i)
        out.append(rr)
    return out


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def download_and_prepare() -> None:
    for name, (url, prefix) in DATASETS.items():
        zip_bytes = _download_bytes(url)
        tmp_dir = DATA_DIR / f"_{name}_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            members = z.namelist()
            members_l = [m.lower() for m in members]

            def find_member(suffix: str) -> str | None:
                target = (prefix + suffix).lower()
                for m, ml in zip(members, members_l, strict=True):
                    if ml.endswith(target):
                        return m
                return None

            table_a = find_member("tableA.csv")
            table_b = find_member("tableB.csv")
            matches = find_member("matches.csv")
            train = find_member("train.csv")
            valid = find_member("valid.csv")
            test = find_member("test.csv")
            if not table_a or not table_b:
                raise RuntimeError(
                    f"Unexpected zip layout for {name}: missing {prefix}tableA.csv / tableB.csv"
                )

            rows_a = _standardize_rows(_read_csv_from_zip(z, table_a))
            rows_b = _standardize_rows(_read_csv_from_zip(z, table_b))
            if matches:
                gt_rows = _read_csv_from_zip(z, matches)
            else:
                # DeepMatcher-hosted zips often provide train/valid/test labeled pairs instead of matches.csv.
                gt_rows = []
                for m in (train, valid, test):
                    if m:
                        gt_rows.extend(_read_csv_from_zip(z, m))

        # Ground truth format: id_a, id_b, label (0/1)
        gt_out: list[dict[str, Any]] = []
        for row in gt_rows:
            keys = {k.lower(): k for k in row.keys()}
            a_key = keys.get("ltable_id") or keys.get("id_a") or keys.get("a_id")
            b_key = keys.get("rtable_id") or keys.get("id_b") or keys.get("b_id")
            label_key = keys.get("label")
            if not a_key or not b_key:
                # fall back to first two columns
                cols = list(row.keys())
                a_key, b_key = cols[0], cols[1]
            label = int(row[label_key]) if label_key else 1
            gt_out.append({"id_a": str(row[a_key]), "id_b": str(row[b_key]), "label": label})

        # Persist in both JSON and CSV for easy CLI use.
        _write_json(DATA_DIR / f"{name}_a.json", rows_a)
        _write_json(DATA_DIR / f"{name}_b.json", rows_b)
        _write_csv(DATA_DIR / f"{name}_a.csv", rows_a)
        _write_csv(DATA_DIR / f"{name}_b.csv", rows_b)

        # Labels CSV
        labels_path = DATA_DIR / f"{name}_labels.csv"
        with labels_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id_a", "id_b", "label"])
            w.writeheader()
            w.writerows(gt_out)

    print(f"Prepared datasets in {DATA_DIR}")


if __name__ == "__main__":
    download_and_prepare()
