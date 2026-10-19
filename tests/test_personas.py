import random
import tempfile
import unittest

from trustpanel.llm import LLM
from trustpanel.personas import clean_text, normalize_weights, segment_topic, unsupported_citations, write_persona

from .fakes import FakeClient

LEAK = ["leaked through overnight", "leaks at night ruined the sheets", "not absorbent enough, leaked", "leaked again"]
SOFT = ["so soft on sensitive skin", "gentle and soft, no rash", "soft material, no irritation", "soft and comfortable"]


def corpus(n=200, seed=1):
    rng = random.Random(seed)
    reviews = []
    for i in range(n):
        leak = i % 2 == 0
        words = rng.choice(LEAK if leak else SOFT)
        reviews.append({"id": f"r{i}", "rating": rng.choice([1, 2] if leak else [4, 5]),
                        "title": "Leaks" if leak else "Soft", "text": f"{words} {rng.choice(['pads', 'briefs'])}"})
    return reviews


class SegmentTest(unittest.TestCase):
    def test_separates_needs_and_keeps_shares(self):
        segments = segment_topic("feminine_care", corpus(), max_k=2)
        self.assertEqual(len(segments), 2)
        self.assertAlmostEqual(sum(s.share for s in segments), 1.0)
        themes = [" ".join(s.top_terms) for s in segments]
        self.assertTrue(any("leak" in t for t in themes))
        self.assertTrue(any("soft" in t for t in themes))
        leaky = next(s for s in segments if "leak" in " ".join(s.top_terms))
        self.assertGreater(leaky.rating_counts[1] + leaky.rating_counts[2], leaky.rating_counts[4] + leaky.rating_counts[5])

    def test_small_topics_get_fewer_segments(self):
        self.assertEqual(len(segment_topic("t", corpus(n=70), max_k=3)), 1)

    def test_deterministic(self):
        a = segment_topic("t", corpus(), max_k=2)
        b = segment_topic("t", corpus(), max_k=2)
        self.assertEqual([s.evidence for s in a], [s.evidence for s in b])


class CleanTextTest(unittest.TestCase):
    def test_strips_html(self):
        self.assertEqual(clean_text("Soft.<br /><br />Strong &amp; cheap").split(), ["Soft.", "Strong", "cheap"])


class PersonaTest(unittest.TestCase):
    def test_invented_citations_are_dropped(self):
        segment = segment_topic("feminine_care", corpus(), max_k=2)[0]
        real = segment.evidence[0]["id"]
        card = {"name": "The heavy-flow worrier", "summary": "s", "situation": "s",
                "needs": [{"need": "No leaks overnight", "evidence": [real, "r99999"]},
                          {"need": "Invented need", "evidence": ["made-up"]}],
                "pain_points": [], "deal_breakers": ["leaks"], "price_sensitivity": "medium",
                "rating_tendency": "harsh"}
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(lambda i, p: card)
            persona = write_persona(LLM(client, "m", tmp), segment)
        self.assertEqual(persona.needs, [{"need": "No leaks overnight", "evidence": [real]}])
        self.assertEqual(unsupported_citations(card, segment), 2)
        self.assertEqual(persona.weight, segment.share)
        self.assertIn("Representative reviews", client.calls[0]["input"])
        self.assertIn("No leaks overnight", persona.card())

    def test_content_filter_drops_blocked_quotes_and_retries(self):
        segment = segment_topic("feminine_care", corpus(), max_k=2)[0]
        segment.evidence[0]["text"] = "BLOCKED quote"
        card = {"name": "n", "summary": "s", "situation": "s", "needs": [], "pain_points": [],
                "deal_breakers": [], "price_sensitivity": "low", "rating_tendency": "fair"}

        def answer(instructions, prompt):
            if "BLOCKED" in prompt:
                raise RuntimeError("Error code: 400 - {'code': 'content_filter'}")
            return card

        with tempfile.TemporaryDirectory() as tmp:
            persona = write_persona(LLM(FakeClient(answer), "m", tmp), segment)
        self.assertEqual(persona.segment["filtered_quotes"], 1)
        self.assertNotIn("BLOCKED", str(persona.segment["evidence"]))

    def test_weights_normalise_within_topic(self):
        class P:
            def __init__(self, pid, w):
                self.persona_id, self.weight = pid, w
        weights = normalize_weights([P("a", 0.2), P("b", 0.6)])
        self.assertAlmostEqual(weights["a"], 0.25)
        self.assertAlmostEqual(weights["b"], 0.75)


if __name__ == "__main__":
    unittest.main()
