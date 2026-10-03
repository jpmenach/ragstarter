"""Answer a question from an index: retrieve chunks, build a grounded prompt, generate.

Each step is explicit (rather than LlamaIndex's one-call query engine) so the pipeline
can stop between retrieval and generation — e.g. to skip the LLM entirely when nothing
relevant was retrieved.
"""

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, ChatMessage

GROUNDING_RULES = """\
Answer the user's question using only the information in the provided context.
If the context does not contain the answer, say that you don't have that information.
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
    text: str
    sources: list[RetrievedChunk]
    # Whether retrieval was relevant enough to answer from. Callers decide what to do
    # when it isn't (e.g. hand off to a human).
    grounded: bool


def answer(
    question: str,
    index: VectorStoreIndex,
    llm: LLM,
    persona: str = "",
    top_k: int = 3,
) -> Answer:
    retrieved = index.as_retriever(similarity_top_k=top_k).retrieve(question)
    # LlamaIndex types the score as optional; vector retrieval always sets it.
    sources = [RetrievedChunk(text=r.get_content(), score=r.score or 0.0) for r in retrieved]

    # Placeholder until a relevance threshold is chosen from measured retrieval scores.
    grounded = True

    response = llm.chat(_build_messages(question, sources, persona))
    return Answer(text=response.message.content.strip(), sources=sources, grounded=grounded)


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
