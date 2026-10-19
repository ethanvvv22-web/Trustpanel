"""A fake Responses client that answers from a function of the prompt."""

import json
from types import SimpleNamespace


class FakeClient:
    def __init__(self, answer):
        self.answer = answer  # (instructions, prompt) -> dict
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, model, reasoning, instructions, input, text):
        self.calls.append({"instructions": instructions, "input": input, "schema": text["format"]["schema"]})
        return SimpleNamespace(output_text=json.dumps(self.answer(instructions, input)),
                               usage=SimpleNamespace(input_tokens=100, output_tokens=20))
