"""Markdown summary of a validation run."""

from typing import Dict


LABELS = {"persona_panel": "Persona panel", "persona_calibrated": "Persona panel, calibrated",
          "generic_llm": "Generic LLM", "generic_calibrated": "Generic LLM, calibrated", "topic_prior": "Topic prior"}
ROWS = [
    ("mean_abs_error", "Mean rating error (stars)", "lower"),
    ("wasserstein", "Distribution distance (stars)", "lower"),
    ("negative_share_error", "Error in share of 1-2 star reviews", "lower"),
    ("pairwise_accuracy", "Picks the better product (pairs)", "higher"),
    ("spearman", "Rank correlation with real means", "higher"),
    ("bias", "Bias (predicted - real mean)", "closer to 0"),
    ("spread_ratio", "Spread vs real spread", "closer to 1"),
]


def _cell(summary: Dict, key: str, metric: str) -> str:
    value = summary[key]["metrics"][metric]
    ci = summary[key]["ci95"].get(metric)
    if value != value:
        return "n/a"
    return f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else f"{value:.2f}"


def render_markdown(summary: Dict) -> str:
    PREDICTORS = summary["predictors"]
    lines = [f"Held-out products: {summary['n_products']}. Values with 95% bootstrap intervals.", "",
             "| Metric | " + " | ".join(LABELS[k] for k in PREDICTORS) + " | Better |",
             "|---|" + "---|" * len(PREDICTORS) + "---|"]
    for metric, label, better in ROWS:
        lines.append(f"| {label} | " + " | ".join(_cell(summary, k, metric) for k in PREDICTORS) + f" | {better} |")
    lines += ["", "Mean rating error by topic:", "",
              "| Topic | " + " | ".join(LABELS[k] for k in PREDICTORS) + " |",
              "|---|" + "---|" * len(PREDICTORS)]
    for topic, row in summary["by_topic"].items():
        lines.append(f"| {topic} | " + " | ".join(f"{row[k]:.2f}" for k in PREDICTORS) + " |")
    return "\n".join(lines)
