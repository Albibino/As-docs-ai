FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    # O tiktoken baixa o vocabulário do tokenizer na primeira chamada. Sem um
    # cache fixo, cada boot do container (usuário sem HOME gravável) baixava de
    # novo da internet e o serviço só subia depois disso.
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . \
 && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')" \
 && chmod -R a+rX "$TIKTOKEN_CACHE_DIR"

USER 65532:65532
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health/live',timeout=4)"
CMD ["sh", "-c", "uvicorn tcc_docs_ai.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]
