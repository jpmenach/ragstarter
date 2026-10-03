"""Answer a question from an index: retrieve chunks, build a grounded prompt, generate.

Each step is explicit (rather than LlamaIndex's one-call query engine) so the pipeline
can stop between retrieval and generation — e.g. to skip the LLM entirely when nothing
relevant was retrieved.
"""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, ChatMessage

from ragstarter.core.config import QueryConfig

# Refusals in free text come in endless phrasings ("No tengo esa información", "Lo siento,
# no tengo información sobre..."), which code can't detect reliably. A fixed sentinel can.
NO_ANSWER = "NO_ANSWER"

GROUNDING_RULES = f"""\
Answer the user's question using only the information in the provided context.
If the context does not explicitly answer the question, reply with exactly {NO_ANSWER} \
and nothing else.
If the context does not mention a service, product, or fact, that means it is unknown, \
not that the answer is "no": reply {NO_ANSWER}.
Do not guess and do not use outside knowledge. Be concise."""

# When the context is mostly one language (e.g. a Spanish FAQ), small models drift into
# that language. Tested on llama3.1:8b: a plain "answer in the same language" was ignored
# both in the system prompt and here; it only held once placed last AND naming the
# conflict with the context explicitly.
LANGUAGE_RULE = (
    "IMPORTANT: reply in the language of the question, "
    "even though the context is in another language."
)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float


@dataclass(frozen=True)
class Answer:
    # Empty when not grounded.
    text: str
    sources: list[RetrievedChunk]
    # Whether the documents could answer the question. Callers decide what to do when
    # they couldn't (e.g. hand off to a human).
    grounded: bool


def answer(
    question: str,
    index: VectorStoreIndex,
    llm: LLM,
    persona: str = "",
    config: QueryConfig | None = None,
) -> Answer:
    config = config or QueryConfig()

    retrieved = index.as_retriever(similarity_top_k=config.top_k).retrieve(question)
    # LlamaIndex types the score as optional; vector retrieval always sets it.
    sources = [RetrievedChunk(text=r.get_content(), score=r.score or 0.0) for r in retrieved]

    # Cheap first check: nothing in the documents is even about this topic.
    if not sources or max(chunk.score for chunk in sources) < config.min_score:
        return Answer(text="", sources=sources, grounded=False)

    # Second check, by the LLM: the topic matches, but the context may still not contain
    # the answer. Similarity scores can't tell those apart; reading the context can.
    response = llm.chat(_build_messages(question, sources, persona))
    text = response.message.content.strip()
    if NO_ANSWER in text:
        return Answer(text="", sources=sources, grounded=False)

    return Answer(text=text, sources=sources, grounded=True)


def _build_messages(
    question: str,
    sources: list[RetrievedChunk],
    persona: str,
) -> list[ChatMessage]:
    system = f"{persona}\n\n{GROUNDING_RULES}" if persona else GROUNDING_RULES
    context = "\n\n---\n\n".join(chunk.text for chunk in sources)
    user = f"Context:\n\n{context}\n\nQuestion: {question}\n\n{LANGUAGE_RULE}"

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
