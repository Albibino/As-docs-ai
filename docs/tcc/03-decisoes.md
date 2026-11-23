# Decisões

## D1. Redis 8 Vector Sets como banco vetorial

**Decisão:** usar os Vector Sets do Redis 8 (`VADD`, `VSIM`, `SETATTR`).

**Por quê:** o serviço já dependia de Redis (idempotência), e a regra era não
introduzir ferramentas novas. O Redis 8 traz Vector Sets nativos, e o `redis-py`
já instalado fala os comandos. Com isso não entra pgvector, Qdrant nem outra
peça na infraestrutura.

**Consequência:** a imagem precisa ser `redis:8` ou mais nova. O
`docker-compose.yml` e a documentação de Railway fixam essa versão.

## D2. Índice append-only com identidade por versão

**Decisão:** cada trecho é o elemento `pageId:version:chunk`, e a versão faz
parte da chave. Um conjunto `…:page:{id}:versions` guarda as versões indexadas,
e o hash `…:page:{id}` guarda `latest`, que só sobe.

**Por quê:** o requisito é "nunca reindexar, só indexar registros novos
versionados, considerando a maior versão como a atualizada". Com a versão na
chave, gravar a v2 não toca os elementos da v1. Uma versão já presente no
conjunto é recusada antes de gerar embeddings (sem custo), e uma versão menor
que `latest` é ignorada.

**Consequência:** o índice cresce a cada versão. Isso é intencional: o
histórico fica preservado, e a busca filtra pela versão atual. Limpeza de
versões antigas é uma decisão futura e está fora do escopo.

## D3. Quem decide a "versão" é o app

**Decisão:** a versão vem do app (`pages.content_version`, incrementada por
trigger no Postgres). O serviço de IA não calcula hash nem data.

**Por quê:** é o app que sabe quando o conteúdo mudou, inclusive pela edição
colaborativa em tempo real. Um contador monotônico deixa a comparação "indexado
< atual" exata e barata.

## D4. Detecção de defasagem feita por código, não pelo modelo

**Decisão:** `ChatService` chama o MCP de forma determinística:
`docs.search_pages` → `docs.get_page_context` → `docs.read_page`. O modelo não
escolhe tools.

**Por quê:** com temperatura 0 e o requisito de nunca responder com dado velho,
deixar a decisão "devo ler a página?" para o modelo seria uma fonte de erro. O
código compara números: se `versão atual > versão indexada`, lê.

**Consequência:** o modelo (quando existir chave) só redige a resposta a partir
da evidência já selecionada.

## D5. A conversa nunca escreve no índice

**Decisão:** o conteúdo lido pelo MCP é ranqueado em memória e descartado
depois.

**Por quê:** o índice tem um único escritor (o indexador do app). Se a conversa
também gravasse, haveria corrida entre os dois e o "append-only" deixaria de ser
verificável.

## D6. Temperatura 0 travada na configuração

**Decisão:** `openai_temperature: float = Field(0.0, ge=0.0, le=0.0)`.
Qualquer outro valor derruba a inicialização.

**Por quê:** é requisito do TCC, não preferência. Modelos de raciocínio da
OpenAI só aceitam `temperature` com `reasoning.effort=none`, por isso esse é o
padrão. Deixar `OPENAI_REASONING_EFFORT` vazio omite o parâmetro.

## D7. "Não encontrado" como evento, não como texto livre

**Decisão:**
- Sem evidência acima de `AI_RAG_MIN_SCORE`, o serviço nem chama o modelo:
  emite `not_found {similar}`.
- Com evidência, o modelo deve responder `NAO_ENCONTRADO` quando o contexto não
  responder à pergunta. O serviço retém o texto enquanto ele ainda pode ser o
  começo dessa sentinela e, se ela se confirmar, troca pelo evento.

**Por quê:** a interface mostra "não encontrei" e as semelhantes de forma
estruturada, e nada inventado chega ao usuário, nem por streaming parcial.

## D8. Provedores sem chave (`hashing` e `extractive`)

**Decisão:**
- `HashingEmbeddingProvider`: vetor normalizado de palavras com hashing, em 256
  dimensões.
- `ExtractiveAnswerProvider`: a resposta é a cópia dos trechos mais
  relevantes, com `[n]`.

**Por quê:** o projeto ainda não tem chave de API e não pode usar credenciais
de outros projetos. Os dois provedores mantêm o fluxo completo funcionando e
testável, e por construção não conseguem inventar nada.

**Consequência:** a qualidade semântica da busca é inferior à dos embeddings da
OpenAI, que dependem de vocabulário compartilhado. Os limiares padrão
(`0.66`/`0.58`) foram calibrados nos testes. Ao trocar de provedor, o modelo
entra na chave do índice (`tcc-docs-ai:{ambiente}:vindex:{modelo}:{tenant}`), então os vetores de
modelos diferentes nunca se misturam e é preciso indexar de novo no modelo novo.

## D9. Vocabulário do tokenizer embutido na imagem

**Achado:** `TokenCounter` (`tokens.py`) carrega o tokenizer do modelo
(`o200k_base` para `gpt-5.2`) na criação do app. O `tiktoken` baixa esse
vocabulário da internet na primeira vez, e o container (usuário sem `HOME`
gravável) baixava de novo a cada boot. O serviço levava de dezenas de segundos
a vários minutos para responder, conforme a rede, e não subia sem internet.

**Decisão:** o `Dockerfile` define `TIKTOKEN_CACHE_DIR=/opt/tiktoken` e
carrega o vocabulário durante o build.

**Consequência:** o boot passou a levar cerca de 2,5 s, inclusive sem acesso à
internet (verificado com o container numa rede `--internal`). Trocar
`OPENAI_MODEL` por um modelo de outra família de tokenizer exige pré-carregar
o vocabulário correspondente no `Dockerfile`.

## D10. gpt-6-luna como modelo de chat

**Verificado na API** antes de configurar:
- o modelo existe na conta;
- `temperature=0` só é aceito com `reasoning.effort=none`. Sem o `reasoning`,
  a API devolve 400 ("temperature is not supported with this model"). É
  exatamente o que `sampling_parameters` já envia;
- o `tiktoken` ainda não conhece o nome. O `TokenCounter` mapeia o modelo para
  `o200k_base`, cuja contagem bateu com `responses.input_tokens.count` da API
  (diferença fixa de 6 tokens de envelope por mensagem). Esse vocabulário já é
  o embutido na imagem (D9).

Embeddings: `text-embedding-3-small`, 1536 dimensões. A troca de `hashing` para
`openai` criou um índice novo e vazio (D8), preenchido pelo indexador do app.

## D11. Regras de prompt calibradas no modelo real

**Achado:** temperatura 0 reduz a variação, mas não a elimina. Com o prompt
original, a mesma pergunta às vezes recebia só `[1]` como resposta e, com duas
páginas divergentes, o modelo oscilava entre listar as duas, escolher uma ou
responder `NAO_ENCONTRADO`.

**Decisão:** três regras novas no `CHAT_SYSTEM_PROMPT`:
- resposta em texto completo, nunca só citações;
- "vale o `mcp_read`" só entre trechos da MESMA página;
- páginas diferentes com respostas diferentes: dizer o que cada uma afirma,
  pelo título, sem escolher e sem `NAO_ENCONTRADO`.

**Medição** (`scripts/chat_variance_probe.py`, 12 execuções por cenário):

| Cenário | Antes | Depois |
|---|---|---|
| Uma página | 2 respostas distintas, metade sem a frase completa | 1 resposta, sempre completa |
| Duas páginas divergentes | 4 distintas, incluindo `[1]` e `NAO_ENCONTRADO` | 2 distintas, que diferem só na posição das citações |
| Sem resposta | `NAO_ENCONTRADO` sempre | `NAO_ENCONTRADO` sempre |

**Consequência:** o E2E não exige texto idêntico, só o fato certo vindo da
página certa (por título ou pela citação `[n]` do evento `sources`).

## D12. Escrita assistida calibrada no modelo real

**Achados** (testando no editor e com `scripts/writing_variance_probe.py`, 10
chamadas por cenário):
1. Pergunta que não pede alteração ("Qual é a capital da França?") depois de
   uma correção aceita: em 8 de 10 vezes o modelo respondia em prosa ("Não há
   informações no documento…"). O serviço rejeitava (`invalid_model_proposal`) e
   o editor mostrava "Algo deu errado".
2. Pedido com fato ausente da página ("acrescente o horário do expediente"): em
   6 de 10 vezes o modelo devolvia uma "correção" do parágrafo em vez de não
   fazer nada.
3. Qualquer regra nova escrita como "não produza nada" fazia o `gpt-6-luna`
   vazar o token de controle do template (`</|assistant|>`, `</|im_end|>`) depois
   da operação certa, repetido até estourar `max_output_tokens`: até 7 de 10
   correções simples falhavam.
4. Às vezes a mesma operação vinha duas vezes.

**Decisões:**
- Em vez de "saída vazia", o prompt pede uma linha explícita
  `{"action":"none","reason":"..."}`. O serviço converte a sentinela em
  `done {operationCount: 0, noChangeReason}`; o modelo nunca escreve no
  documento e o motivo vai para o usuário.
- O serviço trata uma linha de token de controle (`<...|>`) como fim da
  proposta e fecha o stream da OpenAI, sem ler o resto.
- Operações idênticas repetidas são emitidas uma vez só.
- Resposta em prosa continua sendo erro (`invalid_model_proposal`).

**Medição** (pelo `GenerationService`, 10 chamadas por cenário):

| Cenário | Antes | Depois |
|---|---|---|
| Corrigir ortografia | 10/10 certas (prompt original) | 10/10 certas |
| Pergunta de conhecimento geral | vazio | 10/10 "nada", com motivo |
| Pergunta após uma correção | 8/10 erro (prosa) | 10/10 "nada", com motivo |
| Fato ausente da página | 6/10 alteravam o parágrafo | 10/10 "nada", com motivo |
| Transformar em lista | 10/10 certas | 10/10 certas |

Cobertura: `tests/test_service_model_output.py` reproduz cada saída real vista
(tokens de controle, sentinela, duplicata, prosa).

## D13. Tenant = workspace

O `tenantId` é o `workspaceId` do app e isola chaves, busca e estados. O TCC tem
um único workspace, mas o isolamento foi mantido porque já fazia parte da
arquitetura.

## D14. OAuth Client Credentials local

**Decisão:** a IA recebe os tokens técnicos em modo OAuth opcional. Valida
`RS256` com o JWKS configurado, sem introspecção. Só o indexador pode gravar e
consultar o índice; só o assistente pode chamar chat e escrita. Cada rota exige
seu escopo e o workspace cadastrado para a conta de serviço. HMAC continua como
padrão para os ambientes existentes; os dois modos não coexistem. A credencial
delegada do MCP não mudou. Ver [contrato OAuth](06-contratos.md#oauth-20-client-credentials-local).

**Dependência:** o dono autorizou apenas a declaração direta de
`PyJWT[crypto]>=2.14,<3` para essa validação. O pacote já era transitivo do MCP.
`mcp.server.auth.provider.TokenVerifier` é uma interface para receber um
validador; não faz validação por JWKS. O lock reutiliza a entrada existente de
`PyJWT==2.14.0` e o extra `crypto`; nenhum pacote travado mudou de versão. A
imagem verificada contém `PyJWT==2.15.1`, pois o `Dockerfile` instala com
`pip install .`, sem usar o lock. A suíte foi executada com ambas as versões,
com chaves efêmeras e sem o `.env` do projeto.

**Operação:** o JWKS fica em cache por processo e a atualização é limitada
mesmo quando chegam chaves desconhecidas. Se não houver uma chave válida para
verificar o token, a chamada é recusada. A prontidão não depende de uma consulta
ao provedor. A configuração do realm e o corte do Compose ficam no app; os
testes do serviço não substituem o teste ponta a ponta com o Keycloak.
