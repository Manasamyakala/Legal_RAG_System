"""Step 6: GENERATE -- the LLM reads the retrieved chunks and answers.

The prompt template below is the single most important piece of this whole
project for correctness. Without an explicit instruction to answer *only*
from the provided context, the LLM will happily fall back on its own
general knowledge when the documents don't contain the answer -- which
looks like a working RAG system until you ask it something the documents
don't cover and it confidently hallucinates instead of saying "I don't
know." See the README's "How to Verify the LLM Uses Your Documents" section
for exactly how to catch this.
"""

from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the context below, taken from the user's own documents.

Rules:
- If the answer is contained in the context, answer it directly and concisely.
- If the answer is NOT contained in the context, say exactly: "I don't know based on the provided documents."
- Do not use any knowledge you have that isn't in the context below, even if you're confident about it.

Context:
{context}

Question: {question}

Answer:"""


def get_llm(model_spec, temperature=0):
    """Build a LangChain chat model from a "<provider>/<model>" spec, e.g.
    "openai/gpt-4o-mini" or "ollama/llama3". A bare model name with no
    provider prefix (e.g. "gpt-4o-mini") is treated as OpenAI.

    Ollama needs no API key -- it's a locally running model server -- which
    is the whole point of supporting it: this project works with zero paid
    API calls if you don't want to use OpenAI at all.
    """
    if model_spec.startswith("ollama/"):
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_spec.split("/", 1)[1], temperature=temperature)

    model_name = model_spec.split("/", 1)[1] if model_spec.startswith("openai/") else model_spec
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model_name, temperature=temperature)


def build_qa_chain(llm, retriever):
    """Wire retrieval + the prompt + generation into one chain. chain_type
    "stuff" means "stuff all retrieved chunks directly into the prompt" --
    the simplest strategy, and the right one as long as k * chunk_size
    comfortably fits in the model's context window."""
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
