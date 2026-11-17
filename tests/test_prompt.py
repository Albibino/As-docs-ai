from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from tcc_docs_ai.prompt import SYSTEM_PROMPT, build_openai_input
from tcc_docs_ai.schemas import GenerationRequest


def request_model(**overrides: object) -> GenerationRequest:
    payload: dict[str, object] = {
        "tenantId": "t1",
        "sessionId": "s1",
        "request": "edite",
        "blocks": [{"blockId": "b1", "type": "paragraph", "hash": "h1", "content": "original"}],
    }
    payload.update(overrides)
    return GenerationRequest.model_validate(payload)


def page_snapshot(request: GenerationRequest) -> dict:
    snapshot = build_openai_input(request)[1]["content"]
    return json.loads(snapshot.removeprefix("PAGE_SNAPSHOT\n"))


def test_space_pages_reach_the_page_snapshot() -> None:
    request = request_model(spacePages=[{"pageId": "p2", "title": "Arquitetura"}])
    assert page_snapshot(request)["spacePages"] == [{"pageId": "p2", "title": "Arquitetura"}]


def test_space_pages_default_to_an_empty_list() -> None:
    assert page_snapshot(request_model())["spacePages"] == []


def test_space_page_entries_reject_unknown_fields() -> None:
    # Nothing beyond pageId and title may ride along — content especially.
    with pytest.raises(ValidationError):
        request_model(spacePages=[{"pageId": "p2", "title": "T", "content": "leak"}])


def test_system_prompt_documents_the_page_mention_shape() -> None:
    # The client can only build a real page link from a mention node carrying a
    # pageId taken from spacePages; a drifting prompt silently breaks the graph.
    assert '"type":"mention"' in SYSTEM_PROMPT
    assert '"entityType":"page"' in SYSTEM_PROMPT
    assert "spacePages" in SYSTEM_PROMPT
