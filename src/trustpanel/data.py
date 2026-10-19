"""Load and split the Amazon "Health and Personal Care" reviews for hygiene categories.

The split is by product: test products are never seen while building personas,
so a persona's rating of a test product is a genuine out-of-sample prediction.
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

# Hygiene and personal-care categories close to Essity's business: (include, exclude)
# patterns on the product title; first matching topic wins. Generic words such as
# "pads" or "wipes" alone matched nose pads, heating pads and phone cases, so each
# topic requires a specific product term and excludes accessories.
TOPICS = {
    "feminine_care": (
        r"sanitary (pads?|napkins?)|maxi pads?|menstrual (pads?|cups?|discs?)|panty ?liners?|tampons?"
        r"|period (underwear|panties)|feminine (pads?|wash|wipes|hygiene)|postpartum pads?",
        r"heating|nose|knee|makeup|cleaning|scour|sterili[sz]er|steamer|cleaner"),
    "baby_care": (
        r"baby (wipes|diapers?|lotion|wash|shampoo|oil|powder|cream)|diaper (rash|cream)"
        r"|(cloth|disposable|newborn|swim) diapers?|\bdiapers?, size",
        r"bag|caddy|cake|organizer|pail|holder|phone|case\b|safety pins?|for adults|sprayer|hose|deodorant"),
    "wound_care": (
        r"adhesive bandages?|elastic bandages?|bandages?\b|wound (care|dressings?|closure)|gauze"
        r"|blister (pads?|bandages?|plasters?)|hydrocolloid|first aid tape|butterfly closures?",
        r"cast cover|cast protector|nintendo|mario|costume|toy|shears|scissors|clips?\b|camo|hunting"
        r"|\bcats?\b|\bdogs?\b|\bpets?\b|horse|eye patch"),
    "bath_tissue": (
        r"facial tissues?|toilet paper|bath tissue|kleenex",
        r"holder|dispenser|cover|stand|organizer|wedding|party|trump|novelty|prank|gag gift"),
    "kitchen_paper": (
        r"paper towels?|paper napkins?",
        r"holder|dispenser|cover|stand|organizer|plates|wedding|party|novelty"
        r"|dish ?cloths?|replacement"),
    "skin_care": (
        r"(body|hand|face|foot|baby|skin) (lotion|cream|moisturi[sz]er)|moisturi[sz]ing (cream|lotion)"
        r"|barrier cream|skin protectant|healing ointment|dry skin",
        r"warmer|steamer|sprayer|mister|container|bottles?|socks|massage|brush|pain relief|menthol|camphor"
        r"|sleeves?|brace|\bhair\b|cold sore|\blips?\b|sting|bite|face wash|cleanser"),
}
# Incontinence products are out of scope; they are excluded even where a title also
# mentions, say, wipes or skin cream.
OUT_OF_SCOPE = (r"incontinen|adult diapers?|bladder (control|leak)|underpads?|protective underwear"
                r"|adult (briefs?|pull-?ups?)|chux|bed pads?|for adults")
STARS = (1, 2, 3, 4, 5)


@dataclass
class Product:
    product_id: str
    topic: str
    title: str
    brand: str
    price: Optional[float]
    features: List[str]
    description: str
    rating_counts: Dict[int, int] = field(default_factory=lambda: {s: 0 for s in STARS})

    @property
    def n_reviews(self) -> int:
        return sum(self.rating_counts.values())

    @property
    def mean_rating(self) -> float:
        return sum(s * c for s, c in self.rating_counts.items()) / max(self.n_reviews, 1)

    def distribution(self) -> List[float]:
        n = max(self.n_reviews, 1)
        return [self.rating_counts[s] / n for s in STARS]

    def concept(self, max_chars: int = 700) -> str:
        """What a panel sees: the product as described, without brand names or reviews.

        Brand names are removed because the model may remember how well-known
        products were rated, which would make a good prediction meaningless.
        """
        parts = [self.title] + [f"- {f}" for f in self.features[:5]]
        if self.description:
            parts.append(self.description)
        text = "\n".join(p for p in parts if p)
        text = strip_brand(text, self.brand)
        if self.price is not None:
            text += f"\nPrice: ${self.price:.2f}"
        return text[:max_chars]

    def to_json(self) -> dict:
        d = asdict(self)
        d["rating_counts"] = {str(k): v for k, v in self.rating_counts.items()}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Product":
        d = dict(d)
        d["rating_counts"] = {int(k): v for k, v in d["rating_counts"].items()}
        return cls(**d)


@dataclass
class Review:
    product_id: str
    rating: int
    title: str
    text: str


def strip_brand(text: str, brand: str) -> str:
    if not brand or len(brand) < 3:
        return text
    return re.sub(re.escape(brand), "[brand]", text, flags=re.IGNORECASE)


def topic_of(title: str) -> Optional[str]:
    t = title.lower()
    if re.search(OUT_OF_SCOPE, t):
        return None
    for topic, (include, exclude) in TOPICS.items():
        if re.search(include, t) and not re.search(exclude, t):
            return topic
    return None


def may_be_hygiene(title: str) -> bool:
    """Include patterns only: a superset of `topic_of`, used when downloading so that
    exclusion rules can be refined later without downloading the source again."""
    t = title.lower()
    return any(re.search(include, t) for include, _ in TOPICS.values())


def _price(value) -> Optional[float]:
    try:
        return float(str(value).replace("$", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_products(meta_path: str) -> Dict[str, Product]:
    """Products whose title falls in one of the hygiene topics."""
    products = {}
    for m in read_jsonl(meta_path):
        topic = topic_of(m.get("title") or "")
        if topic is None:
            continue
        description = " ".join(m.get("description") or [])
        products[m["parent_asin"]] = Product(
            product_id=m["parent_asin"], topic=topic, title=m.get("title") or "",
            brand=(m.get("store") or "").strip(), price=_price(m.get("price")),
            features=list(m.get("features") or []), description=description[:1000],
        )
    return products


def load_reviews(reviews_path: str, products: Dict[str, Product]) -> Dict[str, List[Review]]:
    """Reviews of the given products; also fills in each product's rating counts."""
    by_product: Dict[str, List[Review]] = defaultdict(list)
    for r in read_jsonl(reviews_path):
        product = products.get(r["parent_asin"])
        if product is None:
            continue
        rating = int(r["rating"])
        product.rating_counts[rating] += 1
        by_product[r["parent_asin"]].append(Review(r["parent_asin"], rating, r.get("title") or "", r.get("text") or ""))
    return dict(by_product)


def _stable_order(product_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{product_id}".encode()).hexdigest()


def split_products(products: Iterable[Product], per_topic: int = 10, min_reviews: int = 30,
                   seed: str = "trustpanel") -> List[str]:
    """Pick test products: per topic, up to `per_topic` products with enough reviews for a
    reliable rating distribution, chosen by a seeded hash so the split is reproducible."""
    by_topic: Dict[str, List[Product]] = defaultdict(list)
    for p in products:
        if p.n_reviews >= min_reviews:
            by_topic[p.topic].append(p)
    test = []
    for topic in TOPICS:
        candidates = sorted(by_topic.get(topic, []), key=lambda p: _stable_order(p.product_id, seed))
        test += [p.product_id for p in candidates[:per_topic]]
    return test
