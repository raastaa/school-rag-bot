import asyncio
import gigachat_client


def test_director_answer_prompt(monkeypatch):
    captured = {}

    async def fake_chat(prompt, timeout=60):
        captured['prompt'] = prompt
        return "ok"

    monkeypatch.setattr(gigachat_client, 'chat', fake_chat)
    snippets = [
        {
            'text': 'Приказ вступает в силу с момента подписания.',
            'source': {'title': 'Приказы', 'page_from': 1, 'page_to': 2, 'heading_path': 'Документооборот'},
        }
    ]
    asyncio.run(
        gigachat_client.answer_from_director_snippets(
            'как правильно составить приказ', snippets, mode='detailed'
        )
    )
    prompt = captured['prompt']
    assert 'НЕЛЬЗЯ использовать внешние знания' in prompt
    assert 'питание' not in prompt
    assert 'Структура приказа' in prompt
