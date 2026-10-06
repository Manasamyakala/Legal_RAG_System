"""Step 4: INDEX -- store chunk vectors in FAISS, saved to disk for reuse.

Why cache to disk at all: embedding every chunk of every document is the
slowest part of setup (each chunk is a small local model inference call).
Without a cache, every single run -- including just asking one more
question -- would re-embed the entire document set from scratch. Caching
means the expensive work happens once per document set, not once per query.
"""

import os

from langchain_community.vectorstores import FAISS

from src.chunker import chunk_documents
from src.document_loader import load_documents


def build_or_load_vector_store(docs_dir, index_dir, embedder):
    """Load a cached FAISS index from index_dir if one exists; otherwise
    load + chunk + embed the documents in docs_dir and build a fresh index.

    `allow_dangerous_deserialization=True` is required by FAISS.load_local
    because the index is loaded via pickle -- safe here since it's an index
    this same codebase wrote, not one downloaded from somewhere else.
    """
    if os.path.isdir(index_dir) and os.listdir(index_dir):
        return FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)

    documents = load_documents(docs_dir)
    if not documents:
        raise RuntimeError(
            f"No documents were loaded from {docs_dir}.\n"
            "Only .pdf, .txt, and .docx files are supported -- add some and try again."
        )

    chunks = chunk_documents(documents)
    store = FAISS.from_documents(chunks, embedder)
    store.save_local(index_dir)
    return store
