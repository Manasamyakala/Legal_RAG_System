"""Step 2: CHUNK -- split large documents into ~500-char overlapping pieces.

Why chunk at all: an LLM prompt has a finite size, and stuffing an entire
50-page handbook into every question would be slow, expensive, and bury the
one relevant paragraph in noise. Why *overlapping* chunks: without overlap,
a sentence that straddles a chunk boundary gets cut in half in both pieces,
and neither half carries the full idea. See the README's "Beginner Tips"
section for what happens when chunk_size is too large or too small.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_documents(documents, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split Documents into overlapping chunks, preserving each chunk's
    source metadata (filename, page number) for later citation."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraph, then line, then word boundaries before
        # falling back to a hard character cut -- keeps chunks readable.
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
