import tempfile
import unittest

from trustpanel.llm import LLM

from .fakes import FakeClient

SCHEMA = {"type": "object", "properties": {"rating": {"type": "integer"}}, "required": ["rating"],
          "additionalProperties": False}


class LLMTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.client = FakeClient(lambda instructions, prompt: {"rating": 4})
        self.llm = LLM(self.client, "gpt-5-mini", self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_answer_is_cached(self):
        self.assertEqual(self.llm.json("be a shopper", "rate it", SCHEMA, "r"), {"rating": 4})
        self.assertEqual(self.llm.json("be a shopper", "rate it", SCHEMA, "r"), {"rating": 4})
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual((self.llm.usage.calls, self.llm.usage.cached), (1, 1))
        self.assertEqual(self.llm.usage.input_tokens, 100)

    def test_samples_are_separate_calls(self):
        self.llm.json("be a shopper", "rate it", SCHEMA, "r", sample=0)
        self.llm.json("be a shopper", "rate it", SCHEMA, "r", sample=1)
        self.assertEqual(len(self.client.calls), 2)

    def test_cache_survives_a_new_instance(self):
        self.llm.json("x", "y", SCHEMA, "r")
        again = LLM(FakeClient(lambda i, p: {"rating": 1}), "gpt-5-mini", self.tmp.name)
        self.assertEqual(again.json("x", "y", SCHEMA, "r"), {"rating": 4})

    def test_retries_then_succeeds(self):
        failures = [RuntimeError("429")]

        def flaky(instructions, prompt):
            if failures:
                raise failures.pop()
            return {"rating": 2}

        llm = LLM(FakeClient(flaky), "m", self.tmp.name, retries=2)
        import trustpanel.llm as module
        sleep, module.time.sleep = module.time.sleep, lambda s: None
        try:
            self.assertEqual(llm.json("a", "b", SCHEMA, "r"), {"rating": 2})
        finally:
            module.time.sleep = sleep


if __name__ == "__main__":
    unittest.main()
