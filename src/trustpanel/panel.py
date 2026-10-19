"""Ask a panel to rate a product concept, and turn the answers into a rating distribution.

Three predictors are compared on the same held-out products:
- persona panel: each of the topic's personas rates the concept `samples` times; the
  ratings are mixed by the share of consumers each persona represents;
- generic LLM: the same model with no persona, given as many draws as the panel;
- topic prior: the rating distribution of the topic's other products, ignoring the concept.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

from .data import STARS, Product
from .llm import LLM, ContentFiltered
from .personas import Persona, normalize_weights

RATING_SCHEMA = {
    "type": "object",
    "properties": {
        "would_buy": {"type": "boolean"},
        "rating": {"type": "integer", "enum": list(STARS)},
        "reason": {"type": "string", "description": "One or two sentences in your own voice."},
    },
    "required": ["would_buy", "rating", "reason"],
    "additionalProperties": False,
}

TASK = """\
Below is a product you are considering, described the way an online shop would describe it.
Imagine you bought it and used it the way you normally would. Give the star rating (1-5) you
would actually leave in an online review, and say why in your own words. Stay in character."""

GENERIC = "You are a typical online shopper buying {topic} products."


def concept_prompt(product: Product) -> str:
    return f"{TASK}\n\nProduct:\n{product.concept()}"


@dataclass
class Answer:
    source: str          # persona id or "generic"
    sample: int
    would_buy: bool
    rating: int
    reason: str


def histogram(ratings: List[int]) -> List[float]:
    n = max(len(ratings), 1)
    return [sum(1 for r in ratings if r == s) / n for s in STARS]


def ask_persona(llm: LLM, persona: Persona, product: Product, sample: int) -> Answer:
    a = llm.json(persona.card(), concept_prompt(product), RATING_SCHEMA, "rating", sample=sample)
    return Answer(persona.persona_id, sample, a["would_buy"], a["rating"], a["reason"])


def ask_generic(llm: LLM, product: Product, sample: int) -> Answer:
    instructions = GENERIC.format(topic=product.topic.replace("_", " "))
    a = llm.json(instructions, concept_prompt(product), RATING_SCHEMA, "rating", sample=sample)
    return Answer("generic", sample, a["would_buy"], a["rating"], a["reason"])


def panel_distribution(answers: List[Answer], personas: List[Persona]) -> List[float]:
    """Mix each persona's rating histogram by the share of consumers it represents."""
    answered = {a.source for a in answers}
    personas = [p for p in personas if p.persona_id in answered]  # all answers filtered: no vote
    weights = normalize_weights(personas)
    mixed = [0.0] * len(STARS)
    for p in personas:
        hist = histogram([a.rating for a in answers if a.source == p.persona_id])
        mixed = [m + weights[p.persona_id] * h for m, h in zip(mixed, hist)]
    return mixed


def topic_priors(products: List[Product]) -> Dict[str, List[float]]:
    """Rating distribution of all reviews per topic; pass products outside the test set."""
    totals: Dict[str, List[int]] = {}
    for p in products:
        acc = totals.setdefault(p.topic, [0] * len(STARS))
        for i, s in enumerate(STARS):
            acc[i] += p.rating_counts[s]
    return {t: [c / max(sum(acc), 1) for c in acc] for t, acc in totals.items()}


def predict(llm: LLM, products: List[Product], personas: List[Persona], priors: Dict[str, List[float]],
            samples: int = 3, workers: int = 8) -> List[Dict]:
    """Run every panel call for every product (in parallel) and return one record per product."""
    jobs = []
    for product in products:
        panel = [p for p in personas if p.topic == product.topic]
        for p in panel:
            jobs += [("persona", product, p, s) for s in range(samples)]
        jobs += [("generic", product, None, s) for s in range(samples * len(panel))]

    def run(job):
        kind, product, persona, sample = job
        try:
            answer = ask_persona(llm, persona, product, sample) if kind == "persona" else ask_generic(llm, product, sample)
        except ContentFiltered:
            answer = None  # counted per product, so blocked calls are visible in the results
        return product.product_id, answer

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, jobs))

    records = []
    for product in products:
        own = [a for pid, a in results if pid == product.product_id]
        answers = [a for a in own if a is not None]
        panel = [p for p in personas if p.topic == product.topic]
        persona_answers = [a for a in answers if a.source != "generic"]
        generic_answers = [a for a in answers if a.source == "generic"]
        records.append({
            "product_id": product.product_id,
            "topic": product.topic,
            "n_reviews": product.n_reviews,
            "truth": product.distribution(),
            "persona_panel": panel_distribution(persona_answers, panel),
            "generic_llm": histogram([a.rating for a in generic_answers]),
            "topic_prior": priors[product.topic],
            "would_buy": {"persona_panel": _share([a.would_buy for a in persona_answers]),
                          "generic_llm": _share([a.would_buy for a in generic_answers])},
            "filtered_calls": len(own) - len(answers),
            "answers": [a.__dict__ for a in answers],
        })
    return records


def _share(flags: List[bool]) -> float:
    return sum(flags) / max(len(flags), 1)
