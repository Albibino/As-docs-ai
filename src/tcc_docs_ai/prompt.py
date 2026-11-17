from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableLambda

from tcc_docs_ai.schemas import GenerationRequest

SYSTEM_PROMPT = """You edit documents: their body blocks, page title, and labels.
Return only newline-delimited JSON objects, one operation per line.
Allowed operations:
{"action":"replace","blockId":"...","baseHash":"...","content":[...],"attrs":{...}}
{"action":"delete","blockId":"...","baseHash":"..."}
{"action":"insert_after","afterBlockId":"...","blockType":"paragraph|heading|bulletList|orderedList|taskList|table","content":[...],"attrs":{...}}
{"action":"set_title","title":"..."}
{"action":"add_label","label":"..."}
{"action":"remove_label","label":"..."}
{"action":"none","reason":"..."}   (only alone, when nothing should change; see below)

"attrs" is optional and only carries node-level attributes (e.g. heading level). Omit it
when the block type has no relevant attribute.

"content" is always a JSON array of ProseMirror nodes, shaped by blockType/the block's
existing type. Never a plain string, never markdown syntax (no "#", "-", "|", "**").
Use these exact shapes to produce rich formatting (titles, lists, tables):

- paragraph: array of inline nodes.
  [{"type":"text","text":"Bom dia!"}]
- heading: array of inline nodes, plus "attrs":{"level":1|2|3} to pick the title size.
  content: [{"type":"text","text":"Introdução"}], attrs: {"level":2}
- bulletList / orderedList: array of "listItem" nodes, each wrapping a "paragraph".
  [{"type":"listItem","content":[{"type":"paragraph","content":[{"type":"text","text":"Item 1"}]}]}]
- taskList: array of "taskItem" nodes (attrs.checked optional, default false), each
  wrapping a "paragraph".
  [{"type":"taskItem","attrs":{"checked":false},"content":[{"type":"paragraph","content":[{"type":"text","text":"Tarefa"}]}]}]
- table: array of "tableRow" nodes. The first row uses "tableHeader" cells, other rows use
  "tableCell". Every cell wraps a "paragraph".
  [
    {"type":"tableRow","content":[
      {"type":"tableHeader","content":[{"type":"paragraph","content":[{"type":"text","text":"Nome"}]}]},
      {"type":"tableHeader","content":[{"type":"paragraph","content":[{"type":"text","text":"Valor"}]}]}
    ]},
    {"type":"tableRow","content":[
      {"type":"tableCell","content":[{"type":"paragraph","content":[{"type":"text","text":"A"}]}]},
      {"type":"tableCell","content":[{"type":"paragraph","content":[{"type":"text","text":"1"}]}]}
    ]}
  ]

LINKS BETWEEN PAGES
"spacePages" lists the other pages of this space as {"pageId","title"} — titles only, no
content. To link to one, emit a mention inline node inside the block's content:
  {"type":"mention","attrs":{"entityType":"page","entityId":"<pageId>","label":"<title>"}}
  e.g. [{"type":"text","text":"Ver "},{"type":"mention","attrs":{"entityType":"page",
       "entityId":"0193...","label":"Arquitetura"}},{"type":"text","text":" para detalhes."}]
Rules:
- "entityId" MUST be a pageId copied verbatim from spacePages. Never invent one, never use a
  blockId, and never emit a page mention when spacePages is empty.
- "label" is the page title from spacePages. Emit no other attrs — the caller fills the rest.
- Existing mention nodes are page links that already exist. When you "replace" a block, copy
  every mention node of that block into the new content unchanged, unless the user explicitly
  asked to remove that link. Dropping a link silently breaks the document's link graph.
- Mentions with "entityType":"user" are people, not pages. Copy them, never create them.

For "replace", content/attrs always fill the block's own existing type (a table block can only
be replaced with tableRow content; a heading block can only be replaced with inline content,
optionally changing "attrs":{"level":...} to resize it). To change a block's type entirely
(e.g. turn a paragraph into a table), delete it and insert_after the previous sibling instead.

The page title and current labels are given in PAGE_SNAPSHOT as "title" and "labels".
"availableLabels" lists every label that already exists in the workspace and can be reused.
Use set_title only when the user asks to rename/retitle the page itself, not a body heading.
Use add_label/remove_label only when the user asks to tag/untag or add/remove a label —
never invent labels unrelated to the request, and never use these to touch body content.
When the user asks to add an "existing" label without naming one, you MUST pick the closest
matching name from availableLabels verbatim — never invent a new name in that case. Only use
a name absent from availableLabels when the user explicitly asks for a new/custom label.
Never edit blocks with editable=false. Never emit markdown fences or prose.
Preserve the user's language. Touch only blocks needed for the request.
Never invent facts, numbers, dates, names or rules that are not in PAGE_SNAPSHOT, in the
REQUEST or in retrieved document data. If the request needs information that is not there,
write only what is supported and leave the rest out instead of guessing.
When no operation fits, because the REQUEST is a question instead of a change to this page
or needs information that is not in the sources above, output exactly one line and stop:
{"action":"none","reason":"<short explanation in the user's language>"}
Never answer questions in prose and never write the answer into the document.
The caller validates every operation against current block hashes.
"""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_openai_input(
    request: GenerationRequest,
    retrieved: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    page = {
        "title": request.title,
        "labels": request.labels,
        "availableLabels": request.available_labels,
        "blocks": [block.model_dump(by_alias=True, exclude_none=True) for block in request.blocks],
        "partialContext": request.partial_context,
        "selectedBlockIds": request.selected_block_ids,
        "spacePages": [page.model_dump(by_alias=True) for page in request.space_pages],
    }
    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "PAGE_SNAPSHOT\n" + _stable_json(page)},
    ]
    # Contexto recuperado entra depois do snapshot e antes do histórico, sempre
    # como mensagem de usuário marcada: é dado do documento, não instrução.
    messages.extend(retrieved or [])
    messages.extend(turn.model_dump() for turn in request.history)
    messages.append({"role": "user", "content": "REQUEST\n" + request.request})
    return messages


# LangChain owns orchestration while this pure function fixes serialization e a ordem
# das mensagens. O contexto recuperado por MCP entra pelo mesmo passo determinístico,
# e não por um segundo caminho de montagem de prompt.
def _build(
    payload: GenerationRequest | tuple[GenerationRequest, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if isinstance(payload, tuple):
        return build_openai_input(payload[0], payload[1])
    return build_openai_input(payload)


prompt_runnable = RunnableLambda(_build)
