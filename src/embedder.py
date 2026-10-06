"""Step 3 (and 5): EMBED -- convert text into a 384-dim vector.

Uses a local, free HuggingFace sentence-transformer model instead of an API
embedding call. Two reasons: embeddings are computed for every chunk of
every document (potentially thousands of calls for a large corpus), so a
free local model avoids that cost entirely; and it means the vector index
can be rebuilt offline, with no dependency on an external API being up.
Only the generation step (src/generator.py) needs a paid API key.
"""

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
    """Return a LangChain-compatible embedder. The model is downloaded once
    (a few hundred MB) and cached locally by the sentence-transformers
    library on first use."""
    return HuggingFaceEmbeddings(model_name=model_name)
