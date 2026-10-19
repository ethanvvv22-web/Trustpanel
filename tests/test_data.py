import json
import os
import tempfile
import unittest

from trustpanel.data import Product, load_products, load_reviews, may_be_hygiene, split_products, strip_brand, topic_of


class TopicTest(unittest.TestCase):
    CASES = {
        "Always Maxi Pads Size 1, Regular, Unscented, 48 Count": "feminine_care",
        "Scott ComfortPlus Toilet Paper, 30 Double Rolls": "bath_tissue",
        "Kleenex Ultra Soft Facial Tissues, 8 Boxes": "bath_tissue",
        "Bounty Select-A-Size Paper Towels, 12 Double Rolls": "kitchen_paper",
        "Vanity Fair Everyday Paper Napkins, 300 Count": "kitchen_paper",
        "Pampers Baby Wipes Sensitive, 576 Count": "baby_care",
        "Band-Aid Brand Flexible Fabric Adhesive Bandages, 100 ct": "wound_care",
        "CeraVe Moisturizing Cream for Dry Skin, 16 oz": "skin_care",
        # Generic words that used to be misclassified.
        "Heating Pad for Back Pain Relief": None,
        "Silicone Nose Pads for Glasses": None,
        "Ever Ready Bandage Shears 7 1/4": None,
        "Baby Diaper Safety Pin for Adults": None,
        "Paper Towel Holder Stainless Steel": None,
        "Self Adhering Bandage Wrap for Cats": None,
        "WINNERS Men's Underwear Soft Cotton Boxer Brief": None,
        "ROSA RUGOSA Menstrual Cup Sterilizer, Steamer Cleaner 3-in-1": None,
        "Eden Shampoo Sink Sprayer Hose for Washing Hair, Baby Wash Sprayer": None,
        "Kids Eye Patches - Fun Boys Design - Latex Free Bandages": None,
        "Swedish Dish Cloths for Kitchen - Reusable Paper Towels": None,
        "Gel Elbow Sleeves Breathable Brace for Dry Skin": None,
        "African Pride Olive Miracle Hair Moisturizing Lotion": None,
        # Incontinence products are out of scope.
        "TENA Incontinence Pads for Women, Heavy, 72 Count": None,
        "Depend FIT-FLEX Incontinence Underwear for Men, Maximum Absorbency": None,
        "Medline Barrier Cream Cloths for Incontinence Care, 24 ct": None,
        "Luna Cup Menstrual Cups Set of 2 Period Cups": "feminine_care",
    }

    def test_known_titles(self):
        for title, topic in self.CASES.items():
            self.assertEqual(topic_of(title), topic, title)

    def test_download_filter_is_a_superset(self):
        for title, topic in self.CASES.items():
            if topic is not None:
                self.assertTrue(may_be_hygiene(title), title)
        self.assertTrue(may_be_hygiene("Paper Towel Holder Stainless Steel"))  # refined away later


def product(pid, topic="bath_tissue", counts=None, brand="Scott"):
    p = Product(product_id=pid, topic=topic, title=f"{brand} Toilet Paper {pid}", brand=brand, price=9.5,
                features=["2-ply", "septic safe"], description="Soft and strong.")
    if counts:
        p.rating_counts.update(counts)
    return p


class ProductTest(unittest.TestCase):
    def test_distribution_and_mean(self):
        p = product("a", counts={5: 6, 1: 2, 3: 2})
        self.assertEqual(p.n_reviews, 10)
        self.assertAlmostEqual(p.mean_rating, (30 + 2 + 6) / 10)
        self.assertEqual(p.distribution(), [0.2, 0.0, 0.2, 0.0, 0.6])

    def test_concept_hides_brand_and_ratings(self):
        concept = product("a", counts={5: 10}).concept()
        self.assertNotIn("Scott", concept)
        self.assertIn("[brand] Toilet Paper", concept)
        self.assertIn("Price: $9.50", concept)
        self.assertNotIn("5", concept.replace("9.50", ""))  # no rating information leaks

    def test_strip_brand_ignores_short_names(self):
        self.assertEqual(strip_brand("A pack", "A"), "A pack")

    def test_json_round_trip(self):
        p = product("a", counts={4: 3})
        self.assertEqual(Product.from_json(json.loads(json.dumps(p.to_json()))), p)


class SplitTest(unittest.TestCase):
    def test_reproducible_and_needs_enough_reviews(self):
        products = [product(f"p{i}", counts={5: 40}) for i in range(30)] + [product("tiny", counts={5: 3})]
        first = split_products(products, per_topic=5, min_reviews=30)
        self.assertEqual(first, split_products(products, per_topic=5, min_reviews=30))
        self.assertEqual(len(first), 5)
        self.assertNotIn("tiny", first)
        self.assertNotEqual(first, split_products(products, per_topic=5, min_reviews=30, seed="other"))


class LoadTest(unittest.TestCase):
    def test_load_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta = os.path.join(tmp, "meta.jsonl")
            reviews = os.path.join(tmp, "reviews.jsonl")
            with open(meta, "w", encoding="utf-8") as f:
                f.write(json.dumps({"parent_asin": "A", "title": "Kleenex Facial Tissues, 6 boxes", "store": "Kleenex",
                                    "price": "$12.99", "features": ["3-ply"], "description": ["Soft"]}) + "\n")
                f.write(json.dumps({"parent_asin": "B", "title": "Heating Pad", "store": "X"}) + "\n")
            with open(reviews, "w", encoding="utf-8") as f:
                for pid, rating in (("A", 5), ("A", 2), ("B", 4)):
                    f.write(json.dumps({"parent_asin": pid, "rating": rating, "title": "t", "text": "x"}) + "\n")
            products = load_products(meta)
            by_product = load_reviews(reviews, products)
        self.assertEqual(list(products), ["A"])
        self.assertEqual(products["A"].price, 12.99)
        self.assertEqual(products["A"].rating_counts[5], 1)
        self.assertEqual(products["A"].rating_counts[2], 1)
        self.assertEqual(len(by_product["A"]), 2)
        self.assertNotIn("B", by_product)


if __name__ == "__main__":
    unittest.main()
