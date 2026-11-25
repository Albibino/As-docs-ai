from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+")


@dataclass(frozen=True)
class Chunk:
    """Trecho indexável de uma versão de página.

    `index` é a posição na versão; a identidade no índice é
    `pageId:version:index`, então duas versões nunca compartilham elemento e
    uma versão indexada nunca é sobrescrita.
    """

    index: int
    heading_path: tuple[str, ...]
    text: str


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Quebra um parágrafo grande por sentença e, em último caso, por tamanho."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    pieces: list[str] = []
    current = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(text: str, *, title: str, max_chars: int) -> list[Chunk]:
    """Corta o Markdown da página em fronteiras de heading e parágrafo.

    Cada trecho carrega o título da página e o caminho de headings, para que um
    trecho isolado ainda diga de onde veio. Determinístico: o mesmo texto gera
    sempre os mesmos trechos, na mesma ordem.
    """
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    path: list[tuple[int, str]] = []
    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if buffer:
            paragraph = " ".join(line.strip() for line in buffer).strip()
            if paragraph:
                paragraphs.append(paragraph)
            buffer.clear()

    def flush_section() -> None:
        flush_paragraph()
        if paragraphs:
            sections.append((tuple(name for _, name in path), list(paragraphs)))
            paragraphs.clear()

    for raw_line in text.splitlines():
        heading = _HEADING.match(raw_line)
        heading_title = raw_line[heading.end():].strip() if heading else ""
        if heading_title:
            flush_section()
            level = len(heading.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, heading_title))
            # O heading também é conteúdo: perguntas citam nomes de seção.
            paragraphs.append(heading_title)
            continue
        if not raw_line.strip():
            flush_paragraph()
        else:
            buffer.append(raw_line)
    flush_section()

    chunks: list[Chunk] = []
    for heading_path, section_paragraphs in sections:
        prefix = " > ".join((title, *heading_path)) if title else " > ".join(heading_path)
        budget = max(max_chars - len(prefix) - 1, max_chars // 2)
        current = ""
        for paragraph in section_paragraphs:
            for piece in _split_long(paragraph, budget):
                if current and len(current) + 1 + len(piece) > budget:
                    chunks.append(Chunk(len(chunks), heading_path, f"{prefix}\n{current}".strip()))
                    current = piece
                else:
                    current = f"{current}\n{piece}".strip()
        if current:
            chunks.append(Chunk(len(chunks), heading_path, f"{prefix}\n{current}".strip()))

    if not chunks and title.strip():
        # Página só com título ainda precisa ser encontrável pelo nome.
        chunks.append(Chunk(0, (), title.strip()))
    return chunks
