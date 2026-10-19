"""Build the processed dataset: hygiene products, the held-out test set and the insight corpus.

    python scripts/prepare_data.py

Writes data/processed/{products.jsonl, test_products.json, calibration_products.json,
insight_reviews.jsonl}. Insight reviews come only from products outside the test and
calibration sets.
"""

import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from trustpanel.data import OUT_OF_SCOPE, load_products, load_reviews, split_products  # noqa: E402

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
MIN_INSIGHT_CHARS = 80  # very short reviews ("great!") say little about needs


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    products = load_products(os.path.join(RAW, "meta_Health_and_Household.hygiene.jsonl"))
    reviews = load_reviews(os.path.join(RAW, "Health_and_Household.hygiene.jsonl"), products)
    products = {pid: p for pid, p in products.items() if p.n_reviews > 0}
    test_ids = split_products(products.values())
    calibration_ids = split_products([p for pid, p in products.items() if pid not in set(test_ids)],
                                     seed="calibration")
    test_set = set(test_ids) | set(calibration_ids)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "products.jsonl"), "w", encoding="utf-8") as f:
        for p in products.values():
            f.write(json.dumps(p.to_json(), ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "test_products.json"), "w", encoding="utf-8") as f:
        json.dump(test_ids, f, indent=1)
    with open(os.path.join(OUT, "calibration_products.json"), "w", encoding="utf-8") as f:
        json.dump(calibration_ids, f, indent=1)
    insight = 0
    with open(os.path.join(OUT, "insight_reviews.jsonl"), "w", encoding="utf-8") as f:
        for pid, rs in reviews.items():
            if pid in test_set:
                continue
            for r in rs:
                # Reviews that bring up incontinence use are left out too (out of scope).
                if len(r.text) >= MIN_INSIGHT_CHARS and not re.search(OUT_OF_SCOPE, f"{r.title} {r.text}".lower()):
                    f.write(json.dumps({"product_id": pid, "topic": products[pid].topic, "rating": r.rating,
                                        "title": r.title, "text": r.text}, ensure_ascii=False) + "\n")
                    insight += 1

    print(f"hygiene products with reviews: {len(products):,}")
    print("by topic:", dict(Counter(p.topic for p in products.values())))
    print(f"test products: {len(test_ids)} -> by topic:", dict(Counter(products[i].topic for i in test_ids)))
    print("test reviews per product (min/median/max):",
          sorted(products[i].n_reviews for i in test_ids)[0],
          sorted(products[i].n_reviews for i in test_ids)[len(test_ids) // 2],
          sorted(products[i].n_reviews for i in test_ids)[-1])
    print(f"calibration products: {len(calibration_ids)}")
    print(f"insight reviews (>= {MIN_INSIGHT_CHARS} chars, other products): {insight:,}")


if __name__ == "__main__":
    main()
