"""Model factories. Embeddings and chat LLM clients are created lazily and cached per process."""
from functools import lru_cache
import time

from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, EventStreamError
from langchain_aws import BedrockEmbeddings, ChatBedrockConverse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from filing_analyst.config import AWS_REGION, CHAT_MODEL_ID, EMBEDDING_MODEL_ID

_BEDROCK_RETRY_CONFIG = Config(
    retries={"max_attempts": 6, "mode": "adaptive"},
    read_timeout=120,
)


class MultiAgentChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse tailored for Claude 5.x in multi-agent LangGraph workflows:

    1. Strips signed `reasoning_content` / `thinking` blocks from prior `AIMessage`s so
       handoffs across agents with different system prompts/tools do not trigger a
       signature `ValidationException`.
    2. Ensures the message list sent to Bedrock never ends with an `AIMessage` (assistant
       prefill is disallowed by Claude 5.x) when a sub-agent returns control to `supervisor`.
    """

    @staticmethod
    def _prepare_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
        cleaned: list[BaseMessage] = []
        for m in messages:
            if isinstance(m, AIMessage) and isinstance(m.content, list):
                new_blocks = [
                    b for b in m.content
                    if not (isinstance(b, dict) and b.get("type") in {"reasoning_content", "thinking"})
                ]
                if len(new_blocks) != len(m.content):
                    m = m.model_copy(update={"content": new_blocks})
            cleaned.append(m)

        if cleaned and isinstance(cleaned[-1], AIMessage):
            sender = getattr(cleaned[-1], "name", None) or "sub-agent"
            cleaned.append(
                HumanMessage(
                    content=f"[{sender} has completed its task above. Proceed with the next step or provide the final answer to the user.]"
                )
            )
        return cleaned

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prepared = self._prepare_messages(messages)
        for attempt in range(1, 4):
            try:
                return super()._generate(prepared, stop=stop, run_manager=run_manager, **kwargs)
            except (EventStreamError, ClientError, BotoCoreError):
                if attempt == 3:
                    raise
                time.sleep(1.5 * attempt)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        prepared = self._prepare_messages(messages)
        for attempt in range(1, 4):
            yielded_any = False
            try:
                for chunk in super()._stream(prepared, stop=stop, run_manager=run_manager, **kwargs):
                    yielded_any = True
                    yield chunk
                return
            except (EventStreamError, ClientError, BotoCoreError):
                if yielded_any or attempt == 3:
                    raise
                time.sleep(1.5 * attempt)


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the shared Bedrock Titan embeddings client."""
    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        region_name=AWS_REGION,
        config=_BEDROCK_RETRY_CONFIG,
    )
    return embeddings


@lru_cache(maxsize=4)
def get_chat_llm(model_id: str = CHAT_MODEL_ID, temperature: float | None = None, max_tokens: int = 2048):
    """Return a cached Bedrock Converse chat model (supports both .invoke() and .stream())."""
    kwargs: dict = {
        "model": model_id,
        "region_name": AWS_REGION,
        "max_tokens": max_tokens,
        "config": _BEDROCK_RETRY_CONFIG,
    }
    if temperature is not None and "claude-sonnet-5" not in model_id:
        kwargs["temperature"] = temperature
    return MultiAgentChatBedrockConverse(**kwargs)
