from __future__ import annotations

import tiktoken


class TokenCounter:
    def __init__(self, model: str) -> None:
        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoding = tiktoken.get_encoding("o200k_base")

    def count(self, value: str) -> int:
        return len(self._encoding.encode(value))
