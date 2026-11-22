"""The answer model. `build_llm` picks the adapter named by ANSWER_PROVIDER."""

from atar_ai_agent.config import Settings
from atar_ai_agent.llm.client import LLMClient


def build_llm(settings: Settings) -> LLMClient | None:
    """The configured adapter, or None when its API key is missing."""
    if settings.answer_provider == "openai":
        if settings.openai_api_key is None:
            return None
        import openai

        from atar_ai_agent.llm.openai_client import OpenAILLMClient

        return OpenAILLMClient(
            openai.AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value()),
            settings.openai_model,
            settings.openai_reasoning_effort,
            settings.answer_max_tokens,
        )
    if settings.anthropic_api_key is None:
        return None
    import anthropic

    from atar_ai_agent.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient(
        anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value()),
        settings.anthropic_model,
        settings.anthropic_effort,
        settings.answer_max_tokens,
    )
