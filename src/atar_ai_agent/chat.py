"""Chat: an answer that remembers the conversation it is part of.

No HTTP here (like answer.py): the route opens the conversation, then turns
these events into Server-Sent Events.

    open_conversation  find the resident's conversation (or start one) and
                       load its recent turns as history
    chat_events        stream the answer, then save the question and answer
                       together once the answer finished cleanly

History is stored on the server and owned by tenant + user. The app sends
only a conversation id, never past turns: a client that could send history
could forge earlier "assistant" answers to steer this one.
"""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from atar_ai_agent.answer import AnswerEvent, Citation, Delta, Done, Failure, Sources, answer_events
from atar_ai_agent.context import RequestContext
from atar_ai_agent.conversations.store import ConversationStore
from atar_ai_agent.conversations.types import CitedSource, Exchange, Owner, StoredMessage
from atar_ai_agent.llm.client import LLMClient
from atar_ai_agent.llm.prompts import HistoryTurn
from atar_ai_agent.rag.protocols import Embedder, VectorStore


@dataclass(frozen=True)
class OpenConversation:
    conversation_id: str
    is_new: bool
    history: list[HistoryTurn]


@dataclass(frozen=True)
class ConversationStarted:
    conversation_id: str
    is_new: bool


type ChatEvent = ConversationStarted | AnswerEvent

HISTORY_NOT_SAVED = {
    "ar": "تعذّر حفظ هذه الرسالة في المحادثة.",
    "en": "This message could not be saved to the conversation.",
}


def owner_of(context: RequestContext) -> Owner:
    return Owner(tenant_id=context.tenant_id, user_id=context.user_id)


def history_from(messages: Sequence[StoredMessage], max_chars: int) -> list[HistoryTurn]:
    """Complete user/assistant pairs, newest kept first, within `max_chars`.

    Pairs only: a question whose answer was never saved would leave the model
    seeing a question with no reply. Oldest pairs are dropped first to fit.
    """
    pairs: list[tuple[HistoryTurn, HistoryTurn]] = []
    for question, answer in pairwise(messages):
        if question.role == "user" and answer.role == "assistant":
            pairs.append(
                (HistoryTurn("user", question.content), HistoryTurn("assistant", answer.content))
            )
    kept: list[tuple[HistoryTurn, HistoryTurn]] = []
    used = 0
    for pair in reversed(pairs):
        size = len(pair[0].text) + len(pair[1].text)
        if used + size > max_chars:
            break
        kept.insert(0, pair)
        used += size
    return [turn for pair in kept for turn in pair]


async def open_conversation(
    conversation_id: str | None,
    context: RequestContext,
    *,
    conversations: ConversationStore,
    max_exchanges: int,
    max_chars: int,
) -> OpenConversation:
    """The resident's conversation with this id and its recent history.

    The app chooses the id ("1234", a UUID, ...). An id this resident has not
    used yet starts a new conversation under it; with no id, one is generated.
    Ids are per resident, so another resident's "1234" is never reached.
    """
    owner = owner_of(context)
    resolved, is_new = await conversations.open(owner, conversation_id)
    if is_new:
        return OpenConversation(resolved, is_new=True, history=[])
    # One extra message, in case the newest stored message is an unpaired question.
    messages = await conversations.messages(resolved, owner, limit=2 * max_exchanges + 1)
    return OpenConversation(resolved, is_new=False, history=history_from(messages or [], max_chars))


async def chat_events(
    question: str,
    context: RequestContext,
    conversation: OpenConversation,
    *,
    conversations: ConversationStore,
    embedder: Embedder,
    store: VectorStore,
    llm: LLMClient,
    top_k: int,
) -> AsyncIterator[ChatEvent]:
    """`conversation` first, then the answer events; the exchange is saved
    just before `done`, and only if the answer finished without an error."""
    yield ConversationStarted(conversation.conversation_id, conversation.is_new)
    query = question
    text: list[str] = []
    citations: list[CitedSource] = []
    failed = False
    async for event in answer_events(
        question,
        context,
        embedder=embedder,
        store=store,
        llm=llm,
        top_k=top_k,
        history=conversation.history,
    ):
        match event:
            case Sources(_, searched):
                query = searched or question
            case Delta(piece):
                text.append(piece)
            case Citation(_, source_key, title, cited_text):
                cited = CitedSource(source_key, title, cited_text)
                if cited not in citations:
                    citations.append(cited)
            case Failure():
                failed = True
            case Done(grounded, language, _, _) if not failed:
                exchange = Exchange(question, query, "".join(text).strip(), citations, grounded)
                try:
                    await conversations.append(
                        conversation.conversation_id, owner_of(context), exchange
                    )
                except Exception:
                    yield Failure("history_not_saved", HISTORY_NOT_SAVED[language])
        yield event
