import time

from src.cost import MemoryCache, ModelRouter, UsageLedger


def test_router_cheap_for_classify():
    r = ModelRouter(cheap="mini", strong="full")
    assert r.pick("classify", "short") == "mini"
    assert r.pick("complex_reason", "x") == "full"
    assert r.pick("chat", "x" * 9000) == "full"


def test_cache_roundtrip_and_ttl():
    c = MemoryCache(ttl_s=1)
    c.set("ns", "prompt", {"a": 1})
    assert c.get("ns", "prompt") == {"a": 1}
    assert c.get("ns", "other") is None
    time.sleep(1.1)
    assert c.get("ns", "prompt") is None


def test_ledger_budget():
    led = UsageLedger()
    led.add("u1", 0.4, tokens=100)
    assert led.allowed("u1", 1.0) is True
    led.add("u1", 0.7)
    assert led.allowed("u1", 1.0) is False
    assert led.usage("u1")["tokens"] == 100


def test_ledger_rejects_negative():
    led = UsageLedger()
    try:
        led.add("u", -1)
        assert False
    except ValueError:
        pass
