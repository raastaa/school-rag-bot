import asyncio
import gigachat_client


def test_self_check_insufficient(monkeypatch):
    async def fake_chat(prompt, timeout=15):
        return "insufficient"

    monkeypatch.setattr(gigachat_client, "chat", fake_chat)
    res = asyncio.run(gigachat_client.self_check_sufficiency("q", ["a"]))
    assert res == "insufficient"
