"""Entry point -- ties all 6 RAG steps together.

    load -> chunk -> embed -> index -> retrieve -> generate

Interactive mode asks questions in a loop; --question answers one question
and exits; --debug prints the retrieved chunks so you can see exactly what
the LLM was given before it answered.
"""

import argparse
import os

from dotenv import load_dotenv

from src.document_loader import load_documents  # noqa: F401  (imported for clarity in --debug hints)
from src.embedder import get_embedder
from src.generator import build_qa_chain, get_llm
from src.retriever import get_retriever
from src.vector_store import build_or_load_vector_store

DEFAULT_DOCS_DIR = "Learnings"
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def answer_question(qa_chain, question, debug=False):
    result = qa_chain.invoke({"query": question})

    if debug:
        print("\n--- Retrieved chunks (this is what the LLM actually saw) ---")
        for doc in result["source_documents"]:
            print(f"[{doc.metadata.get('source', 'unknown')}]")
            print(doc.page_content[:300])
            print("---")

    print(f"\nAnswer: {result['result']}")
    sources = sorted({doc.metadata.get("source", "unknown") for doc in result["source_documents"]})
    print(f"Sources: {', '.join(sources) if sources else '(none retrieved)'}")


def parse_args():
    parser = argparse.ArgumentParser(description="RAG from Scratch -- ask questions about your documents.")
    parser.add_argument("--question", type=str, default=None, help="Ask a single question and exit.")
    parser.add_argument("--debug", action="store_true", help="Print retrieved chunks before the answer.")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve per question (default 3).")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help="e.g. openai/gpt-4o-mini (default) or ollama/llama3 for a local model.",
    )
    parser.add_argument("--docs-dir", type=str, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    using_openai = not args.model.startswith("ollama/")
    if using_openai and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set.\n"
            "  cp .env.example .env\n"
            "  # edit .env and add your key\n"
            "Or use a local model instead, no key needed: --model ollama/llama3"
        )

    embedder = get_embedder()
    vector_store = build_or_load_vector_store(args.docs_dir, args.index_dir, embedder)
    retriever = get_retriever(vector_store, k=args.k)
    llm = get_llm(args.model)
    qa_chain = build_qa_chain(llm, retriever)

    if args.question:
        answer_question(qa_chain, args.question, debug=args.debug)
        return

    print("RAG from Scratch -- ask a question ('exit' or Ctrl+C to quit)\n")
    while True:
        try:
            question = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        answer_question(qa_chain, question, debug=args.debug)


if __name__ == "__main__":
    main()
