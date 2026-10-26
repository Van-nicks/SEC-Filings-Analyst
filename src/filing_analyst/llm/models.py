"""Model factories. Embeddings and chat LLM clients are created lazily and cached per process."""
from functools import lru_cache

from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

from filing_analyst.config import AWS_REGION, CHAT_MODEL_ID, EMBEDDING_MODEL_ID


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the shared Bedrock Titan embeddings client."""
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=AWS_REGION)
    return embeddings


@lru_cache(maxsize=4)
def get_chat_llm(model_id: str = CHAT_MODEL_ID, temperature: float = 0.0, max_tokens: int = 2048):
    """Return a cached Bedrock Converse chat model (supports both .invoke() and .stream())."""
    return ChatBedrockConverse(
        model=model_id,
        region_name=AWS_REGION,
        temperature=temperature,
        max_tokens=max_tokens,
    )
