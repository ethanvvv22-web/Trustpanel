"""Structured-output calls to a model deployed on Microsoft Foundry, cached on disk.

Every response is cached by (model, effort, instructions, input, schema, sample index),
so re-running an experiment costs nothing and reproduces the same answers, while
`sample` lets a caller draw several independent answers to the same prompt.
"""

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


class ResponsesClient(Protocol):
    """The slice of the OpenAI client this module uses."""

    responses: Any


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def hit(self) -> None:
        with self._lock:
            self.cached += 1


class ContentFiltered(Exception):
    """The service's content filter refused the prompt; retrying the same prompt is pointless."""


class LLM:
    def __init__(self, client: ResponsesClient, model: str, cache_dir: str, effort: str = "low",
                 retries: int = 4):
        self.client = client
        self.model = model
        self.effort = effort
        self.cache_dir = cache_dir
        self.retries = retries
        self.usage = Usage()
        os.makedirs(cache_dir, exist_ok=True)

    def json(self, instructions: str, prompt: str, schema: Dict[str, Any], name: str, sample: int = 0) -> Dict[str, Any]:
        """Return the model's answer parsed against a strict JSON schema."""
        key = hashlib.sha256(json.dumps([self.model, self.effort, instructions, prompt, schema, sample],
                                        sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            self.usage.hit()
            with open(path, encoding="utf-8") as f:
                return json.load(f)["answer"]

        for attempt in range(self.retries + 1):
            try:
                response = self.client.responses.create(
                    model=self.model, reasoning={"effort": self.effort}, instructions=instructions, input=prompt,
                    text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
                )
                answer = json.loads(response.output_text)
                break
            except Exception as e:
                if "content_filter" in str(e):
                    raise ContentFiltered(str(e)) from e
                if attempt == self.retries:
                    raise
                time.sleep(2 * (attempt + 1))
        usage = getattr(response, "usage", None)
        self.usage.add(getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + f".{threading.get_ident()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "effort": self.effort, "sample": sample, "answer": answer}, f,
                      ensure_ascii=False)
        os.replace(tmp, path)
        return answer

    def chat(self, instructions: str, messages: List[Dict[str, str]]) -> str:
        """Free-text reply to a conversation ([{"role", "content"}]); not cached."""
        try:
            response = self.client.responses.create(model=self.model, reasoning={"effort": self.effort},
                                                    instructions=instructions, input=messages)
        except Exception as e:
            if "content_filter" in str(e):
                raise ContentFiltered(str(e)) from e
            raise
        usage = getattr(response, "usage", None)
        self.usage.add(getattr(usage, "input_tokens", 0) or 0, getattr(usage, "output_tokens", 0) or 0)
        return response.output_text


def foundry_client(endpoint: str):
    """OpenAI-compatible client for a Foundry project, authenticated with the Azure CLI login."""
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    # The CLI can need well over the default 10 s to refresh a token after the machine wakes.
    project = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential(process_timeout=60))
    return project.get_openai_client()


def from_env(cache_dir: str, effort: Optional[str] = None) -> LLM:
    return LLM(foundry_client(os.environ["AZURE_AI_PROJECT_ENDPOINT"]), os.environ["MODEL_DEPLOYMENT_NAME"],
               cache_dir, effort=effort or os.environ.get("REASONING_EFFORT", "low"))
