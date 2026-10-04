# ragstarter

A local-first RAG engine: point it at a folder of documents and ask questions about them,
answered by an LLM running on your own machine. The engine is business-agnostic; everything
specific to one use case lives in a separate example built on top of it.

The included demo is a FAQ assistant for a fictional veterinary clinic. The FAQ is in
Spanish, and the assistant answers questions asked in either Spanish or English.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Ingestion, retrieval, and grounded answer generation with a local LLM | Done |
| 2 | Detect questions the documents can't answer and hand off to a human instead | Done |
| 3 | Appointment booking via function calling (Google Calendar), in the demo | Next |
| 4 | WhatsApp integration for the demo | Planned |
| 5 | Source citations, conversation memory, usage analytics | Optional |

## How it works

```
Ingestion:  documents ─► split by markdown headers ─► token-bounded chunks ─► bge-m3 embeddings ─► Chroma

Query:      question ─► retrieve top-k chunks ─► grounded prompt ─► local LLM (Ollama) ─► answer + sources
                                │                                         │
                        top score too low?                       replied NO_ANSWER?
                                └──────────────► not grounded ◄───────────┘
```

When the engine returns "not grounded", the demo hands the conversation off to clinic staff
instead of answering.

The code is split in two:

- **`src/ragstarter/core/`**: the reusable engine. It loads a directory of documents with
  LlamaIndex's `SimpleDirectoryReader` (tested so far with Markdown), chunks and embeds them,
  retrieves relevant chunks, and generates an answer constrained to that context. It knows
  nothing about any particular business.
- **`examples/veterinaria-demo/`**: one concrete use of the engine. It holds the clinic's FAQ,
  the assistant's persona, and (in later phases) the clinic-specific actions like booking
  appointments.

The rule for where new code goes: if it would make sense for any document set, it belongs in
`core`; if it only makes sense for a vet clinic, it belongs in the example.

## Design decisions

- **Multilingual embeddings (`BAAI/bge-m3`).** An English-only embedding model would put a
  Spanish FAQ and an English question in unrelated parts of the vector space. bge-m3 maps
  both languages into one space, so an English question retrieves the right Spanish chunk.
- **Chunk by document structure first, size second.** Documents are split on markdown
  headers before being token-limited (500 tokens, 75 overlap), so two unrelated topics never
  share a chunk. Testing showed this mattered: when the FAQ used bold text instead of real
  headers, unrelated policies were merged into one chunk and retrieval got worse.
- **An explicit retrieve → prompt → generate pipeline**, instead of LlamaIndex's one-call
  query engine. Keeping the steps separate means the engine can inspect retrieval quality
  and decide *not* to call the LLM at all.
- **Two checks before trusting an answer.** Similarity scores measure "same topic", not
  "contains the answer": a question about adopting a cat scored 0.45 against the FAQ, higher
  than a question the FAQ does answer (0.37). So a score threshold (0.30) only filters
  clearly off-topic questions, without spending an LLM call on them. Questions that are on
  topic but unanswered are caught by the LLM itself, which is told to reply with a fixed
  `NO_ANSWER` sentinel (easy for code to detect, unlike free-text refusals) and to treat
  "not mentioned" as unknown rather than "no".
- **The engine depends on LlamaIndex's abstract `LLM` type**, built from config by a single
  factory function. Swapping the local model for a hosted API is a config change, not a
  change to the query pipeline.
- **Grounding rules in the engine, persona in the example.** "Answer only from the context"
  applies to any business, so it's in `core`; the clinic's name and tone are passed in by
  the demo.

## Running the demo

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) with the model pulled:

```bash
ollama pull llama3.1:8b
```

Then, from the repository root:

```bash
uv sync
uv run python examples/veterinaria-demo/scripts/ingest.py      # build the vector index
uv run python examples/veterinaria-demo/scripts/ask.py "¿Atienden los domingos?"
uv run python examples/veterinaria-demo/scripts/chat.py        # web chat at http://localhost:7860
uv run python examples/veterinaria-demo/scripts/query_test.py  # run the evaluation
```

`ask.py --sources` and the web chat also show the retrieved chunks and their similarity
scores. The web chat (Gradio) is a testing tool; it shows the conversation, but each message
is still answered on its own. Re-run
`ingest.py` after editing anything in `data/`; it rebuilds the index from scratch.

The first ingestion downloads the bge-m3 embedding model (about 2.3 GB).

Tested on a laptop with 8 GB of RAM available to WSL2 and an RTX 3050 Ti (4 GB VRAM). The
8B model doesn't fit entirely in VRAM, so Ollama splits it between GPU and CPU (roughly
42/58), and answers take 4–10 seconds each.

## Sample output

Real runs of `ask.py`:

```
$ ask.py "¿Cuánto cuesta castrar a mi gato?"
La esterilización de gato o gata cuesta $70 USD.

$ ask.py "¿Atienden los domingos?"
No, la clínica permanece cerrada los domingos.

$ ask.py "What does a general consultation cost?"
A general consultation costs $25 USD.

$ ask.py "Do you do home visits?"          # not in the FAQ: handed off, not guessed
No tengo esa información, pero un miembro de nuestro equipo te responderá pronto.
Si es una emergencia, llama al +57 300 123 4567 (opción 2).
```

## Evaluation

`query_test.py` runs 23 questions in three groups: answerable from the FAQ, on topic but not
in the FAQ, and off topic. A case passes only if the bot makes the right call (answer vs.
hand off) and, when it answers, the answer contains the key facts from the FAQ, e.g. the
right price. Expected facts are regexes, so they accept Spanish or English answers.

Checking facts rather than eyeballing answers mattered: a wrong sterilization price ($130
for a cat, read from the female-dog row of the price table) looked plausible and went
unnoticed until the test checked it.

Current result with `llama3.1:8b`: **19–21 of 23** across runs. All near-miss and
off-topic questions are handed off correctly; the failures are answer-quality issues listed
below. The LLM isn't fully deterministic, so run it more than once before trusting a result.

## Known limitations

All with the local `llama3.1:8b`; comparing against a larger hosted model is planned.

- **Yes/no answers that contradict their own facts.** "¿Abren los festivos?" sometimes gets
  "Sí, la clínica permanece cerrada los festivos": the facts are right, the "Sí" is not.
- **Misreading tables across languages.** "How much is it to spay my female cat?" gets the
  female-dog price ($130) instead of the cat price ($70). The Spanish question is usually
  answered correctly.
- **Answer language.** With an all-Spanish context, English questions are sometimes answered
  in Spanish. A plain "answer in the question's language" instruction got 0 of 4 English
  answers; the current wording, which names the conflict with the context, gets 3 of 4.
- **Source wording matters.** Two misreadings were fixed by rewording the FAQ, not the code:
  "cerrado para consultas programadas, pero contamos con..." was read as *open*, and a price
  row labeled "perra hembra" was matched to "gata". Documents fed to a small model work best
  with one fact per sentence.

## Project layout

```
src/ragstarter/core/
  config.py       ingestion, LLM, and query settings
  ingest.py       load documents, chunk, embed, persist to Chroma
  llm.py          build the configured LLM backend
  query.py        retrieve, check relevance, generate a grounded answer
examples/veterinaria-demo/
  data/faq.md     the clinic's FAQ (Spanish)
  scripts/
    clinic.py     paths, persona, and hand-off message for the clinic
    ingest.py     build the index
    ask.py        ask one question
    chat.py       web chat for manual testing (Gradio)
    query_test.py evaluation (answer vs. hand-off, plus key facts)
```

## License

MIT. See [LICENSE](LICENSE).
