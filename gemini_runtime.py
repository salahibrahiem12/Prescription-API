"""Shared, bounded Gemini access for the single-worker public demo."""
import os
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException

_calls = deque()
_lock = Lock()


def reserve_call():
    """Global budget covers OCR passes and matching fallbacks, including retries."""
    now = time.monotonic()
    with _lock:
        while _calls and now - _calls[0] >= 60:
            _calls.popleft()
        if len(_calls) >= int(os.getenv("GEMINI_CALLS_PER_MINUTE", "10")):
            raise HTTPException(429, "Demo AI limit reached. Please retry in one minute.",
                                headers={"Retry-After": "60"})
        _calls.append(now)
