"""Model factories. The embeddings client is created lazily and cached per process."""
from functools import lru_cache

from langchain_aws import BedrockEmbeddings

from filing_analyst.config import AWS_REGION, EMBEDDING_MODEL_ID


@lru_cache(maxsize=1)
def get_embeddings():
    """Return the shared Bedrock Titan embeddings client."""
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, region_name=AWS_REGION)
    return embeddings
