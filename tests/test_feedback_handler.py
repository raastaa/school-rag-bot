import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import asyncio

# ruff: noqa: E402
os.environ.setdefault("BOT_TOKEN", "123456:ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789")

import bot


def test_feedback_handler_removes_markup(monkeypatch):
    def dummy_log_feedback(qid, score, comment):
        return None

    monkeypatch.setattr(bot, "log_feedback", dummy_log_feedback)

    message = SimpleNamespace(reply_markup=True)

    async def edit_reply_markup():
        message.reply_markup = None

    message.edit_reply_markup = AsyncMock(side_effect=edit_reply_markup)
    message.delete = AsyncMock()

    cb = SimpleNamespace(
        data="fb:1:42",
        message=message,
        answer=AsyncMock(),
    )

    asyncio.run(bot.feedback_handler(cb))

    assert message.reply_markup is None
    cb.answer.assert_called_with("Спасибо")


def test_feedback_handler_error(monkeypatch):
    def faulty_log_feedback(*args, **kwargs):
        raise RuntimeError

    monkeypatch.setattr(bot, "log_feedback", faulty_log_feedback)

    cb = SimpleNamespace(
        data="fb:1:42",
        message=SimpleNamespace(edit_reply_markup=AsyncMock(), delete=AsyncMock()),
        answer=AsyncMock(),
    )

    asyncio.run(bot.feedback_handler(cb))

    cb.answer.assert_called_with("Не удалось сохранить отзыв", show_alert=True)
