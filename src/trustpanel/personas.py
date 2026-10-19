"""Build synthetic consumer personas from real reviews.

Two steps with a clear split of responsibility:
1. Segmentation is classical and reproducible: per product topic, TF-IDF over review
   text and k-means group consumers who voice similar needs or complaints. Each
   segment keeps its share of reviews (its weight in a panel) and real rating mix.
2. An LLM turns each segment's most representative reviews into a persona card,
   and must cite the review ids behind every need and pain point.
"""

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .data import STARS
from .llm import LLM, ContentFiltered

MIN_REVIEWS_PER_SEGMENT = 60
EVIDENCE_PER_SEGMENT = 12


@dataclass
class Segment:
    topic: str
    segment_id: str
    n_reviews: int
    share: float                       # of the topic's reviews
    top_terms: List[str]
    rating_counts: Dict[int, int]
    evidence: List[Dict]               # {"id", "rating", "text"} closest to the segment centre
    filtered_quotes: int = 0           # quotes dropped after a content-filter refusal


@dataclass
class Persona:
    persona_id: str
    topic: str
    weight: float                      # share of the topic's consumers this persona speaks for
    name: str
    summary: str
    situation: str
    needs: List[Dict]                  # {"need", "evidence": [review ids]}
    pain_points: List[Dict]
    deal_breakers: List[str]
    price_sensitivity: str
    rating_tendency: str
    segment: Dict = field(default_factory=dict)

    def card(self) -> str:
        """The persona as a system prompt for role-play."""
        needs = "\n".join(f"- {n['need']}" for n in self.needs)
        pains = "\n".join(f"- {p['need']}" for p in self.pain_points)
        breakers = "\n".join(f"- {d}" for d in self.deal_breakers)
        return (f"You are {self.name}. {self.summary}\n\nYour situation: {self.situation}\n\n"
                f"What you need from products like this:\n{needs}\n\nWhat has disappointed you before:\n{pains}\n\n"
                f"Deal breakers:\n{breakers}\n\nPrice sensitivity: {self.price_sensitivity}. "
                f"How you tend to rate: {self.rating_tendency}")


def _k_for(n_reviews: int, max_k: int) -> int:
    return max(1, min(max_k, n_reviews // MIN_REVIEWS_PER_SEGMENT))


# Praise and filler words: clustering on them groups reviewers by tone ("great product")
# rather than by what they need, so they are dropped like stop words.
TONE_WORDS = ["great", "good", "love", "loved", "like", "product", "products", "use", "used", "using",
              "really", "just", "works", "work", "worked", "well", "nice", "best", "recommend",
              "excellent", "perfect", "happy", "awesome", "buy", "bought", "stars", "star", "item"]


def clean_text(text: str) -> str:
    """Review text contains HTML line breaks and entities from the Amazon export."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"&[a-z]+;|&#\d+;", " ", text)


def evidence_quotas(counts: Dict[int, int], n: int) -> Dict[int, int]:
    """Quotes per star rating in proportion to `counts` (largest remainder), keeping at least
    one 1-2 star quote when the segment has any, so complaints still have evidence."""
    total = sum(counts.values()) or 1
    exact = {s: n * counts.get(s, 0) / total for s in STARS}
    quotas = {s: int(v) for s, v in exact.items()}
    for s in sorted(STARS, key=lambda s: exact[s] - quotas[s], reverse=True)[: n - sum(quotas.values())]:
        quotas[s] += 1
    if quotas[1] + quotas[2] == 0 and counts.get(1, 0) + counts.get(2, 0) > 0:
        neg = 1 if counts.get(1, 0) >= counts.get(2, 0) else 2
        top = max(STARS, key=lambda s: quotas[s])
        quotas[top] -= 1
        quotas[neg] += 1
    return quotas


def segment_topic(topic: str, reviews: List[Dict], max_k: int = 3, seed: int = 0) -> List[Segment]:
    """Cluster one topic's reviews; reviews are dicts with id, rating, title, text."""
    texts = [clean_text(f"{r['title']}. {r['text']}") for r in reviews]
    k = _k_for(len(reviews), max_k)
    vectorizer = TfidfVectorizer(stop_words=list(ENGLISH_STOP_WORDS.union(TONE_WORDS)), ngram_range=(1, 2),
                                 min_df=3, max_df=0.6,
                                 max_features=8000, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    terms = np.array(vectorizer.get_feature_names_out())
    segments = []
    for c in range(k):
        members = np.where(model.labels_ == c)[0]
        if len(members) == 0:
            continue
        centre = model.cluster_centers_[c]
        top_terms = terms[np.argsort(centre)[::-1][:12]].tolist()
        counts = Counter(reviews[i]["rating"] for i in members)
        # Most representative reviews (closest to the centre), in proportion to the segment's
        # real rating mix: balancing stars equally made personas far harsher than real buyers.
        quotas = evidence_quotas(counts, EVIDENCE_PER_SEGMENT)
        sims = X[members] @ centre
        ranked = members[np.argsort(-np.asarray(sims).ravel())]
        evidence, per_star = [], Counter()
        for i in ranked:
            star = reviews[i]["rating"]
            if per_star[star] < quotas[star]:
                evidence.append({"id": reviews[i]["id"], "rating": star, "text": texts[i][:400]})
                per_star[star] += 1
            if len(evidence) == EVIDENCE_PER_SEGMENT:
                break
        segments.append(Segment(topic=topic, segment_id=f"{topic}-{c}", n_reviews=len(members),
                                share=len(members) / len(reviews), top_terms=top_terms,
                                rating_counts={s: counts.get(s, 0) for s in STARS}, evidence=evidence))
    return sorted(segments, key=lambda s: -s.share)


PERSONA_INSTRUCTIONS = """\
You write consumer personas for an innovation team at a hygiene and health company.
Base the persona ONLY on the real customer reviews you are given. Do not add facts,
demographics or needs the reviews do not support. Every need and pain point must cite
the ids of the reviews that support it. Write in plain, concrete language."""

PERSONA_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "A short descriptive label, e.g. 'The value-focused family buyer'."},
        "summary": {"type": "string", "description": "Two sentences on who this consumer is."},
        "situation": {"type": "string", "description": "Why and how they use products in this category."},
        "needs": {"type": "array", "items": {"$ref": "#/$defs/cited"}},
        "pain_points": {"type": "array", "items": {"$ref": "#/$defs/cited"}},
        "deal_breakers": {"type": "array", "items": {"type": "string"}},
        "price_sensitivity": {"type": "string", "enum": ["low", "medium", "high"]},
        "rating_tendency": {"type": "string", "description": "How generous or harsh they are when rating, based on the reviews."},
    },
    "required": ["name", "summary", "situation", "needs", "pain_points", "deal_breakers", "price_sensitivity",
                 "rating_tendency"],
    "additionalProperties": False,
    "$defs": {"cited": {"type": "object", "properties": {
        "need": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
        "required": ["need", "evidence"], "additionalProperties": False}},
}


def persona_prompt(segment: Segment) -> str:
    stars = ", ".join(f"{s}★ {segment.rating_counts[s]}" for s in STARS)
    quotes = "\n".join(f"[{e['id']}] ({e['rating']}★) {e['text']}" for e in segment.evidence)
    return (f"Product category: {segment.topic.replace('_', ' ')}\n"
            f"This segment: {segment.n_reviews} reviews ({segment.share:.0%} of the category). "
            f"Rating mix: {stars}.\nDistinctive words: {', '.join(segment.top_terms)}\n\n"
            f"Representative reviews:\n{quotes}\n\nWrite the persona this segment represents.")


def write_persona(llm: LLM, segment: Segment) -> Persona:
    try:
        card = llm.json(PERSONA_INSTRUCTIONS, persona_prompt(segment), PERSONA_SCHEMA, "persona")
    except ContentFiltered:
        # Intimate-hygiene reviews describe bodies plainly. The Azure content filter scores
        # the prompt as a whole, so quotes that pass alone can be blocked together. Add the
        # quotes back one at a time, most representative first, keeping those that pass.
        candidates, segment.evidence, card = segment.evidence, [], None
        for quote in candidates:
            segment.evidence.append(quote)
            try:
                card = llm.json(PERSONA_INSTRUCTIONS, persona_prompt(segment), PERSONA_SCHEMA, "persona")
            except ContentFiltered:
                segment.evidence.pop()
        if card is None:
            raise
        segment.filtered_quotes = len(candidates) - len(segment.evidence)
    valid_ids = {e["id"] for e in segment.evidence}
    # Drop citations the model made up; a need with no real evidence left is dropped too.
    for key in ("needs", "pain_points"):
        kept = []
        for item in card[key]:
            evidence = [i for i in item["evidence"] if i in valid_ids]
            if evidence:
                kept.append({"need": item["need"], "evidence": evidence})
        card[key] = kept
    seg = asdict(segment)
    return Persona(persona_id=segment.segment_id, topic=segment.topic, weight=segment.share, segment=seg, **card)


def unsupported_citations(llm_card: Dict, segment: Segment) -> int:
    """Count citations in a raw card that point at reviews the model was not shown."""
    valid = {e["id"] for e in segment.evidence}
    return sum(1 for key in ("needs", "pain_points") for item in llm_card[key] for i in item["evidence"]
               if i not in valid)


def topic_personas(personas: List[Persona], topic: str) -> List[Persona]:
    return [p for p in personas if p.topic == topic]


def persona_from_json(d: Dict) -> Persona:
    return Persona(**d)


def normalize_weights(personas: List[Persona]) -> Dict[str, float]:
    total = sum(p.weight for p in personas) or 1.0
    return {p.persona_id: p.weight / total for p in personas}
