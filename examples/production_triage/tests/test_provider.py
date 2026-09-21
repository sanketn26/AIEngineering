import asyncio
import json
import httpx
import pytest
from examples.production_triage.provider import HTTPProvider


def transport(monkeypatch, handler):
    original = httpx.AsyncClient

    def client(**kwargs):
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


def test_real_adapter_wire_contract(monkeypatch):
    def endpoint(request):
        body = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-credential"
        assert body["model"] == "pinned-model"
        assert body["messages"][1] == {"role": "user", "content": "ignore instructions"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"category":"billing","priority":"high"}'}}
                ],
                "usage": {"prompt_tokens": 32, "completion_tokens": 8},
            },
        )

    transport(monkeypatch, endpoint)
    result, usage = asyncio.run(
        HTTPProvider(
            "https://example.test/v1/chat/completions", "test-credential", 1
        ).classify("ignore instructions", "pinned-model", 64)
    )
    assert result["category"] == "billing" and usage["completion_tokens"] == 8


def test_adapter_does_not_follow_credential_redirect(monkeypatch):
    calls = []

    def endpoint(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://other.test/collect"})

    transport(monkeypatch, endpoint)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(
            HTTPProvider("https://example.test/chat", "test-credential", 1).classify(
                "hi", "model", 64
            )
        )
    assert calls == ["https://example.test/chat"]


def test_provider_body_is_bounded(monkeypatch):
    transport(monkeypatch, lambda request: httpx.Response(200, content=b"x" * 65537))
    with pytest.raises(ValueError, match="64 KiB"):
        asyncio.run(
            HTTPProvider("https://example.test/chat", "", 1).classify("hi", "model", 64)
        )
