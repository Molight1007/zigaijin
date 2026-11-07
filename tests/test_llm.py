from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from evo.adapters.deepseek import DEEPSEEK_FLASH
from evo.adapters.llm import ChatClient, LLMError


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Opener:
    def __init__(self, *results):
        self.results = list(results)
        self.requests = []
        self.timeouts = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def chat_payload(content="hello"):
    return {"choices": [{"message": {"content": content}}]}


def client(opener, *, profile=None, env=None, retries=2, delay=0.0):
    return ChatClient(
        profile or DEEPSEEK_FLASH,
        opener=opener,
        env=env if env is not None else {"DEEPSEEK_API_KEY": "secret-value"},
        max_retries=retries,
        retry_delay=delay,
        timeout=30.0,
    )


def test_complete_returns_message_content_and_sends_the_model():
    opener = Opener(Response(chat_payload("hi")))
    text = client(opener).complete([{"role": "user", "content": "hello"}])
    assert text == "hi"
    request = opener.requests[0]
    body = json.loads(request.data.decode("utf-8"))
    assert body["model"] == DEEPSEEK_FLASH.name
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert request.full_url == "https://api.deepseek.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer secret-value"
    assert opener.timeouts == [30.0]


def test_json_mode_sets_response_format():
    opener = Opener(Response(chat_payload()))
    client(opener).complete([{"role": "user", "content": "x"}], json_mode=True)
    body = json.loads(opener.requests[0].data.decode("utf-8"))
    assert body["response_format"] == {"type": "json_object"}


def test_retries_transient_failures_then_succeeds():
    opener = Opener(HTTPError("u", 500, "boom", {}, None), Response(chat_payload("ok")))
    assert client(opener).complete([{"role": "user", "content": "x"}]) == "ok"
    assert len(opener.requests) == 2


def test_auth_failure_is_not_retried():
    opener = Opener(HTTPError("u", 401, "no", {}, None))
    with pytest.raises(LLMError, match="401"):
        client(opener).complete([{"role": "user", "content": "x"}])
    assert len(opener.requests) == 1


def test_network_failure_names_the_cause_without_the_key():
    opener = Opener(URLError("offline"), URLError("offline"), URLError("offline"))
    with pytest.raises(LLMError) as excinfo:
        client(opener).complete([{"role": "user", "content": "x"}])
    assert "offline" in str(excinfo.value)
    assert "secret-value" not in str(excinfo.value)


def test_malformed_response_raises():
    opener = Opener(Response({"choices": []}))
    with pytest.raises(LLMError, match="choices"):
        client(opener).complete([{"role": "user", "content": "x"}])


def test_missing_key_raises_model_error():
    from evo.adapters.deepseek import ModelError

    with pytest.raises(ModelError, match="DEEPSEEK_API_KEY"):
        client(Opener(), env={}).complete([{"role": "user", "content": "x"}])


def test_repr_hides_the_key():
    opener = Opener()
    instance = client(opener)
    assert "secret-value" not in repr(instance)


def test_rejects_an_empty_message_list():
    opener = Opener()
    with pytest.raises(LLMError, match="messages"):
        client(opener).complete([])
