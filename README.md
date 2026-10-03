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
| 2 | Detect low-relevance retrieval and escalate to a human instead of answering | Next |
| 3 | Appointment booking via function calling (Google Calendar), in the demo | Planned |
| 4 | WhatsApp integration for the demo | Planned |
| 5 | Source citations, conversation memory, usage analytics | Optional |

## How it works

```
Ingestion:  documents ─► split by markdown headers ─► token-bounded chunks ─► bge-m3 embeddings ─► Chroma

Query:      question ─► retrieve top-k chunks ─► grounded prompt ─► local LLM (Ollama) ─► answer + sources
```

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
  and decide *not* to call the LLM, which Phase 2 relies on.
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
uv run python examples/veterinaria-demo/scripts/query_test.py  # ask the test questions
```

The first ingestion downloads the bge-m3 embedding model (about 2.3 GB).

Tested on a laptop with 8 GB of RAM available to WSL2 and an RTX 3050 Ti (4 GB VRAM). The
8B model doesn't fit entirely in VRAM, so Ollama splits it between GPU and CPU (roughly
42/58), and answers take 4–10 seconds each.

## Sample output

From `query_test.py`, with the scores of the retrieved chunks shown under each answer:

```
=== ¿Cuánto cuesta esterilizar a mi gata?  (4.4s)
La esterilización de una gata cuesta $130 USD.
  [0.540] ## Precios de referencia  Los siguientes precios son de referencia y pueden...

=== How much notice do you need to cancel an appointment?  (6.0s)
You need to give at least 4 hours of notice to cancel or reprogram an appointment.
  [0.544] ### Citas y cancelaciones  Las citas se pueden agendar por teléfono, correo...

=== ¿Qué debo hacer si mi mascota tiene una emergencia de madrugada?  (10.8s)
Si tu mascota tiene una emergencia de madrugada, llama a la línea de emergencias al
+57 300 123 4567 (opción 2). El equipo estará disponible para ayudarte 24 horas al día,
los 7 días de la semana.
  [0.472] ## Emergencias  Contamos con línea de emergencias disponible 24 horas, los 7 días...
```

## Known limitations

- **Answer language with a small local model.** When the retrieved context is entirely
  Spanish, `llama3.1:8b` sometimes answers an English question in Spanish, especially when a
  FAQ sentence already reads like a finished answer. The current prompt gets 3 of 4 English
  test questions answered in English; a plain "answer in the question's language"
  instruction got 0 of 4.
- **No relevance threshold yet.** The top chunks always go to the LLM, even for an
  off-topic question; only the prompt's instructions keep the model from answering beyond
  the context. Phase 2 adds a check that stops before generation instead.

## Project layout

```
src/ragstarter/core/
  config.py       ingestion and LLM settings
  ingest.py       load documents, chunk, embed, persist to Chroma
  llm.py          build the configured LLM backend
  query.py        retrieve, build the grounded prompt, generate an answer
examples/veterinaria-demo/
  data/faq.md     the clinic's FAQ (Spanish)
  scripts/        ingest.py, query_test.py
```
