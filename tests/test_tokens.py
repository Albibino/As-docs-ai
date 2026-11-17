from __future__ import annotations

from tcc_docs_ai.tokens import TokenCounter


def test_unknown_model_uses_o200k_base() -> None:
    # gpt-6-luna ainda não está no tiktoken; o fallback precisa ser o
    # o200k_base, que é o vocabulário embutido na imagem Docker e cuja
    # contagem confere com `responses.input_tokens.count` da API.
    counter = TokenCounter("gpt-6-luna")
    assert counter._encoding.name == "o200k_base"
    assert counter.count("Política de férias: 45 dias.") > 0


def test_known_model_keeps_its_own_encoding() -> None:
    assert TokenCounter("gpt-5.2")._encoding.name == "o200k_base"
