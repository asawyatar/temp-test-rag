"""The collaborators the HTTP layer needs, in one place."""

from dataclasses import dataclass

from atar_ai_agent.api.auth import IdentityVerifier
from atar_ai_agent.conversations.store import ConversationStore
from atar_ai_agent.llm.client import LLMClient
from atar_ai_agent.rag.protocols import Embedder


@dataclass
class Services:
    """The app's collaborators. Tests pass fakes; None means "build the real one".

    A verifier or LLM that cannot be built (no backend URL, no API key) stays
    None, and the answer endpoint answers 503 until it is configured.
    """

    embedder: Embedder | None = None
    verifier: IdentityVerifier | None = None
    llm: LLMClient | None = None
    conversations: ConversationStore | None = None
