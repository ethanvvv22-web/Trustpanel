"""Predict rating distributions for the held-out products and score them against real ratings.

    python scripts/run_validation.py --name main --samples 3

Panels also rate a separate calibration set; a linear map from predicted to real mean fitted
there gives the calibrated predictors. Writes results/<name>/{config.json, predictions.jsonl,
calibration_predictions.jsonl, calibration.json, summary.json, summary.md}.
"""

import argparse
import json
import os
import sys
from datetime import datetime

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

from trustpanel import llm as llm_module  # noqa: E402
from trustpanel.calibration import calibrate  # noqa: E402
from trustpanel.data import Product, read_jsonl  # noqa: E402
from trustpanel.panel import predict, topic_priors  # noqa: E402
from trustpanel.personas import persona_from_json  # noqa: E402
from trustpanel.report import render_markdown  # noqa: E402
from trustpanel.validation import summarize  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument("--samples", type=int, default=3, help="answers per persona per product")
    parser.add_argument("--effort", default=None, help="reasoning effort (default: REASONING_EFFORT or low)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N test products (smoke test)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(os.path.join(ROOT, ".env"))
    products = {p["product_id"]: Product.from_json(p) for p in read_jsonl(os.path.join(PROCESSED, "products.jsonl"))}
    with open(os.path.join(PROCESSED, "test_products.json"), encoding="utf-8") as f:
        test_ids = json.load(f)[: args.limit]
    with open(os.path.join(PROCESSED, "calibration_products.json"), encoding="utf-8") as f:
        calibration_ids = json.load(f)[: args.limit]
    with open(os.path.join(PROCESSED, "personas.json"), encoding="utf-8") as f:
        personas = [persona_from_json(p) for p in json.load(f)]
    test_set = set(test_ids)
    priors = topic_priors([p for pid, p in products.items() if pid not in test_set])

    llm = llm_module.from_env(os.path.join(ROOT, "data", "cache"), effort=args.effort)
    out = os.path.join(ROOT, "results", args.name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": llm.model, "effort": llm.effort, "samples": args.samples, "products": len(test_ids),
                   "calibration_products": len(calibration_ids),
                   "personas": len(personas), "started": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    records = predict(llm, [products[i] for i in test_ids], personas, priors, samples=args.samples,
                      workers=args.workers)
    calibration_records = predict(llm, [products[i] for i in calibration_ids], personas, priors,
                                  samples=args.samples, workers=args.workers)
    fitted = {key: calibrate(records, calibration_records, key) for key in ("persona_panel", "generic_llm")}
    with open(os.path.join(out, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(fitted, f, indent=2)
    with open(os.path.join(out, "calibration_predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in calibration_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(out, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = summarize(records)
    summary["usage"] = {"calls": llm.usage.calls, "cached": llm.usage.cached,
                        "input_tokens": llm.usage.input_tokens, "output_tokens": llm.usage.output_tokens}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    md = render_markdown(summary)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print("\nCalibration (real mean = intercept + slope * predicted mean):", json.dumps(fitted))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
