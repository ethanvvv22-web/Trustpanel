"""Segment the insight reviews per topic and write one persona per segment.

    python scripts/build_personas.py

Writes data/processed/personas.json (with the review quotes behind each persona; local only,
since review text may not be redistributed) and results/personas.json (cards and statistics
without review text, safe to publish).
"""

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from dotenv import load_dotenv  # noqa: E402

from trustpanel import llm as llm_module  # noqa: E402
from trustpanel.data import read_jsonl  # noqa: E402
from trustpanel.personas import segment_topic, unsupported_citations, write_persona  # noqa: E402

PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv(os.path.join(ROOT, ".env"))
    by_topic = defaultdict(list)
    for i, r in enumerate(read_jsonl(os.path.join(PROCESSED, "insight_reviews.jsonl"))):
        by_topic[r["topic"]].append({"id": f"R{i}", **r})

    segments = []
    for topic, reviews in sorted(by_topic.items()):
        topic_segments = segment_topic(topic, reviews)
        segments += topic_segments
        print(f"{topic}: {len(reviews):,} reviews -> {len(topic_segments)} segments "
              + ", ".join(f"{s.share:.0%} [{' '.join(s.top_terms[:4])}]" for s in topic_segments), flush=True)

    llm = llm_module.from_env(os.path.join(ROOT, "data", "cache"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        personas = list(pool.map(lambda s: write_persona(llm, s), segments))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(PROCESSED, "personas.json"), "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in personas], f, indent=1, ensure_ascii=False)
    public = []
    for p in personas:
        d = asdict(p)
        d["segment"] = {k: v for k, v in d["segment"].items() if k != "evidence"}
        public.append(d)
    with open(os.path.join(RESULTS, "personas.json"), "w", encoding="utf-8") as f:
        json.dump(public, f, indent=1, ensure_ascii=False)

    cited = sum(len(i["evidence"]) for p in personas for k in ("needs", "pain_points") for i in getattr(p, k))
    print(f"\n{len(personas)} personas; {cited} citations kept; tokens in/out "
          f"{llm.usage.input_tokens:,}/{llm.usage.output_tokens:,} ({llm.usage.cached} cached)")
    for p in personas:
        print(f"  {p.persona_id:<18} {p.weight:4.0%}  {p.name}")


if __name__ == "__main__":
    main()
