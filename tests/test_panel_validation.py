import math
import tempfile
import unittest

from trustpanel.data import Product
from trustpanel.llm import LLM
from trustpanel.panel import histogram, panel_distribution, predict, topic_priors
from trustpanel.personas import Persona
from trustpanel.validation import (bootstrap, expected, metrics, negative_share, pairwise_accuracy, spread,
                                   summarize, wasserstein)

from .fakes import FakeClient


def persona(pid, weight, topic="bath_tissue"):
    return Persona(persona_id=pid, topic=topic, weight=weight, name=pid, summary="s", situation="s",
                   needs=[{"need": "soft", "evidence": ["r1"]}], pain_points=[], deal_breakers=[],
                   price_sensitivity="medium", rating_tendency="fair")


def product(pid, counts, topic="bath_tissue"):
    p = Product(product_id=pid, topic=topic, title=f"Tissue {pid}", brand="", price=5.0, features=[], description="")
    p.rating_counts.update(counts)
    return p


class DistributionTest(unittest.TestCase):
    def test_histogram_and_mixing(self):
        self.assertEqual(histogram([5, 5, 1, 3]), [0.25, 0, 0.25, 0, 0.5])
        from trustpanel.panel import Answer
        answers = [Answer("a", 0, True, 5, ""), Answer("a", 1, True, 5, ""), Answer("b", 0, False, 1, "")]
        mixed = panel_distribution(answers, [persona("a", 0.75), persona("b", 0.25)])
        self.assertEqual(mixed, [0.25, 0, 0, 0, 0.75])

    def test_topic_priors(self):
        priors = topic_priors([product("x", {5: 3}), product("y", {1: 1})])
        self.assertEqual(priors["bath_tissue"], [0.25, 0, 0, 0, 0.75])


class MetricTest(unittest.TestCase):
    def test_basic_quantities(self):
        self.assertEqual(expected([0, 0, 0, 0, 1]), 5)
        self.assertEqual(spread([0, 0, 1, 0, 0]), 0)
        self.assertEqual(negative_share([0.1, 0.2, 0.3, 0.2, 0.2]), 0.30000000000000004)
        self.assertAlmostEqual(wasserstein([1, 0, 0, 0, 0], [0, 0, 0, 0, 1]), 4)
        self.assertEqual(wasserstein([0.2] * 5, [0.2] * 5), 0)

    def test_pairwise_accuracy_only_counts_clear_gaps(self):
        good, bad, close = [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 0.1, 0.9]
        records = [{"topic": "t", "truth": good, "p": good}, {"topic": "t", "truth": bad, "p": bad},
                   {"topic": "t", "truth": close, "p": bad}]
        # good vs bad: right (1); good vs close: real gap 0.1 < 0.3, skipped;
        # bad vs close: both predicted 1 star, a tie, counted as half right (0.5).
        self.assertEqual(pairwise_accuracy(records, "p"), 0.75)
        # Products in different topics are never compared.
        self.assertTrue(math.isnan(pairwise_accuracy([{"topic": "a", "truth": good, "p": good},
                                                      {"topic": "b", "truth": bad, "p": bad}], "p")))

    def test_perfect_predictor(self):
        records = [{"topic": "t", "truth": d, "x": d} for d in ([0, 0, 0, 0, 1], [0.5, 0, 0, 0, 0.5], [0, 0, 1, 0, 0])]
        m = metrics(records, "x")
        self.assertEqual((m["mean_abs_error"], m["wasserstein"], m["bias"]), (0, 0, 0))
        self.assertAlmostEqual(m["spread_ratio"], 1)
        ci = bootstrap(records, lambda rs: metrics(rs, "x"), n=50)
        self.assertEqual(ci["mean_abs_error"], [0, 0])


class PredictTest(unittest.TestCase):
    def test_end_to_end_with_fake_model(self):
        # Personas say 5 stars; the generic shopper says 3.
        def answer(instructions, prompt):
            rating = 3 if instructions.startswith("You are a typical") else 5
            return {"would_buy": rating > 3, "rating": rating, "reason": "r"}

        products = [product("p1", {5: 8, 1: 2}), product("p2", {4: 5, 2: 5})]
        personas = [persona("a", 0.6), persona("b", 0.4), persona("other", 1.0, topic="skin_care")]
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeClient(answer)
            records = predict(LLM(client, "m", tmp), products, personas, {"bath_tissue": [0.2] * 5}, samples=2)
        self.assertEqual(len(client.calls), 2 * (2 * 2 + 2 * 2))   # per product: 2 personas x 2 + 4 generic
        r = records[0]
        self.assertEqual(r["persona_panel"], [0, 0, 0, 0, 1])
        self.assertEqual(r["generic_llm"], [0, 0, 1, 0, 0])
        self.assertEqual(r["topic_prior"], [0.2] * 5)
        self.assertEqual(r["would_buy"], {"persona_panel": 1.0, "generic_llm": 0.0})
        self.assertNotIn("Tissue p1 ", " ".join(c["input"] for c in client.calls if "skin" in c["instructions"]))
        summary = summarize(records, n_boot=20)
        self.assertEqual(summary["n_products"], 2)
        self.assertIn("persona_panel", summary["by_topic"]["bath_tissue"])


if __name__ == "__main__":
    unittest.main()
