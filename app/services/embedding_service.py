from openai import OpenAI
import app.core.config as config

EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, cheap and fast
EMBEDDING_DIM = 1536

_client = OpenAI(api_key=config.OPENAI_API_KEY)


def generate_embedding(text: str) -> list[float]:
    """Generate a 1536-dim embedding vector for the given text using OpenAI."""
    text = text.strip().replace("\n", " ")
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts in a single API call (more efficient)."""
    cleaned = [t.strip().replace("\n", " ") for t in texts]
    response = _client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned,
    )
    # Results are returned in the same order as input
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
