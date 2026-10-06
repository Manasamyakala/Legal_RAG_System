"""Step 5: RETRIEVE -- given a question, find the top-k most similar chunks.

Deliberately a one-line wrapper: FAISS's similarity search already does the
real work (embed the query, compare against every stored vector via cosine
similarity, return the closest k). This file exists so main.py and the
generator never have to know it's FAISS underneath -- swapping in a
different vector database later only touches vector_store.py and this file.
"""


def get_retriever(vector_store, k=3):
    """Return a LangChain retriever that returns the k most similar chunks
    to a query. See the README's "What does k mean" section for the
    precision/recall trade-off of raising or lowering k."""
    return vector_store.as_retriever(search_kwargs={"k": k})
