"""Stream the Amazon Reviews 2023 "Health and Household" category, keeping only hygiene products.

    python scripts/download_filtered.py

The full category is ~14 GB; only matching products and their reviews are written to
data/raw/ (a few hundred MB). Downloads resume from the last byte after a network error.
Hou et al. (2024), https://amazon-reviews-2023.github.io/ -- research use only.
"""

import json
import os
import sys
import time
import urllib.request
from typing import Callable, Iterator

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from trustpanel.data import may_be_hygiene  # noqa: E402

BASE = "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw"
CATEGORY = "Health_and_Household"
RAW = os.path.join(ROOT, "data", "raw")
META_OUT = os.path.join(RAW, f"meta_{CATEGORY}.hygiene.jsonl")
REVIEWS_OUT = os.path.join(RAW, f"{CATEGORY}.hygiene.jsonl")
CHUNK = 1 << 20
RETRIES = 8


def stream_lines(url: str) -> Iterator[bytes]:
    """Yield complete lines; after a dropped connection, re-request from the last byte."""
    offset, buffer, failures = 0, b"", 0
    while True:
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        if buffer:
                            yield buffer
                        return
                    offset += len(chunk)
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    yield from lines
        except Exception as e:  # network errors: resume from `offset`
            failures += 1
            if failures > RETRIES:
                raise
            print(f"  connection lost at {offset / 1e6:.0f} MB ({e}); resuming", flush=True)
            time.sleep(5 * failures)


def filter_file(url: str, out_path: str, keep: Callable[[dict], bool], label: str) -> int:
    kept, seen, started = 0, 0, time.time()
    tmp = out_path + ".part"
    with open(tmp, "w", encoding="utf-8") as out:
        for line in stream_lines(url):
            if not line.strip():
                continue
            seen += 1
            record = json.loads(line)
            if keep(record):
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
            if seen % 500_000 == 0:
                print(f"  {label}: {seen:,} read, {kept:,} kept, {time.time() - started:.0f}s", flush=True)
    os.replace(tmp, out_path)
    return kept


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(META_OUT):
        print("Filtering product metadata ...", flush=True)
        n = filter_file(f"{BASE}/meta_categories/meta_{CATEGORY}.jsonl", META_OUT,
                        lambda m: may_be_hygiene(m.get("title") or ""), "meta")
        print(f"kept {n:,} hygiene products", flush=True)
    with open(META_OUT, encoding="utf-8") as f:
        product_ids = {json.loads(line)["parent_asin"] for line in f}
    if not os.path.exists(REVIEWS_OUT):
        print(f"Filtering reviews for {len(product_ids):,} products ...", flush=True)
        n = filter_file(f"{BASE}/review_categories/{CATEGORY}.jsonl", REVIEWS_OUT,
                        lambda r: r.get("parent_asin") in product_ids, "reviews")
        print(f"kept {n:,} reviews", flush=True)
    print("done")


if __name__ == "__main__":
    main()
