from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from evo.adapters.deepseek import ModelProfile


DEFAULT_BASE_URL = "https://api.deepseek.com/v1"


class LLMError(RuntimeError):
    """A chat completion request failed or returned an unusable response."""


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise LLMError("messages must be a sequence of chat messages")
    rows = list(value)
    if not rows:
        raise LLMError("messages must not be empty")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise LLMError(f"message is not an object: {row!r}")
        role = row.get("role")
        content = row.get("content")
        if not isinstance(role, str) or not role.strip():
            raise LLMError(f"message role must be a non-empty string: {role!r}")
        if not isinstance(content, str):
            raise LLMError("message content must be a string")
        normalized.append({"role": role, "content": content})
    return normalized


def _temperature(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMError(f"temperature must be a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 2.0:
        raise LLMError(f"temperature must be in [0, 2]: {value!r}")
    return number


@dataclass(frozen=True)
class ChatClient:
    profile: ModelProfile
    opener: Callable[..., object] = urlopen
    env: Mapping[str, str] | None = None
    timeout: float = 180.0
    max_retries: int = 3
    retry_delay: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ModelProfile):
            raise LLMError("profile must be a ModelProfile")
        if not callable(self.opener):
            raise LLMError("opener must be callable")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)):
            raise LLMError(f"timeout must be a number: {self.timeout!r}")
        if not math.isfinite(float(self.timeout)) or float(self.timeout) <= 0:
            raise LLMError(f"timeout must be positive: {self.timeout!r}")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise LLMError(f"max_retries must be an integer: {self.max_retries!r}")
        if self.max_retries < 0:
            raise LLMError(f"max_retries must be non-negative: {self.max_retries!r}")
        if isinstance(self.retry_delay, bool) or not isinstance(self.retry_delay, (int, float)):
            raise LLMError(f"retry_delay must be a number: {self.retry_delay!r}")
        if float(self.retry_delay) < 0:
            raise LLMError(f"retry_delay must be non-negative: {self.retry_delay!r}")

    def __repr__(self) -> str:
        return (
            f"ChatClient(profile={self.profile.redacted()!r}, "
            f"timeout={self.timeout!r}, max_retries={self.max_retries!r})"
        )

    def _url(self) -> str:
        base = self.profile.base_url or DEFAULT_BASE_URL
        return base.rstrip("/") + "/chat/completions"

    def _request(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
    ) -> Request:
        body: dict[str, object] = {
            "model": self.profile.name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return Request(
            self._url(),
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.profile.api_key(self.env)}",
            },
            method="POST",
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        rows = _messages(messages)
        heat = _temperature(temperature)
        if max_tokens is not None:
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
                raise LLMError(f"max_tokens must be a positive integer: {max_tokens!r}")
        if not isinstance(json_mode, bool):
            raise LLMError(f"json_mode must be a bool: {json_mode!r}")

        request = self._request(
            rows,
            temperature=heat,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )
        attempts = self.max_retries + 1
        last_error = ""
        for attempt in range(attempts):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    raw = response.read()
                return self._content(raw)
            except HTTPError as exc:
                last_error = f"HTTP {exc.code} {exc.reason}"
                if exc.code != 429 and exc.code < 500:
                    raise LLMError(f"chat request failed: {last_error}") from exc
            except URLError as exc:
                last_error = f"network error: {exc.reason}"
            except (OSError, TimeoutError) as exc:
                last_error = f"request error: {exc}"
            if attempt + 1 < attempts and self.retry_delay:
                time.sleep(float(self.retry_delay))
        raise LLMError(f"chat request failed after {attempts} attempts: {last_error}")

    @staticmethod
    def _content(raw: bytes) -> str:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f"chat response is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMError("chat response is not a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("chat response has no choices")
        first = choices[0]
        if not isinstance(first, dict):
            raise LLMError("chat response choice is not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise LLMError("chat response choice has no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("chat response message has no content")
        return content
