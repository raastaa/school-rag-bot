from aiogram import BaseMiddleware
import time


class Throttle(BaseMiddleware):
    def __init__(self, per_user_sec: float = 0.7):
        super().__init__()
        self.last: dict[int, float] = {}
        self.delay = per_user_sec

    async def __call__(self, handler, event, data):
        user = getattr(getattr(event, "from_user", None), "id", None)
        now = time.monotonic()
        if user is not None and now - self.last.get(user, 0.0) < self.delay:
            return
        if user is not None:
            self.last[user] = now
        return await handler(event, data)
