"""Step 1: LOAD -- read files from disk into LangChain Document objects.

Why a separate file for this: loading is the one step that's genuinely
different per file type (a PDF page is not a .txt file is not a .docx
paragraph). Keeping that branching here means every other step in the
pipeline can work with a plain list of Document objects and never has to
know or care what format a chunk originally came from.
"""

import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

# Extension -> loader class. Add a new file type by adding one line here --
# nothing downstream (chunker, embedder, vector store) needs to change.
SUPPORTED_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def load_documents(docs_dir):
    """Load every supported file in docs_dir into a flat list of Documents.

    A PDF becomes one Document per page (so retrieval can later cite a
    specific page); a .txt or .docx file becomes a single Document.
    Unsupported file types (e.g. .csv) are silently skipped -- see the
    README's "How to Add Your Own Documents" table for what's supported.
    """
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(
            f"{docs_dir} does not exist.\n"
            f"  mkdir -p {docs_dir}\n"
            "  # then add your .pdf/.txt/.docx files"
        )

    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        ext = os.path.splitext(filename)[1].lower()
        loader_cls = SUPPORTED_LOADERS.get(ext)
        if loader_cls is None:
            continue
        path = os.path.join(docs_dir, filename)
        loader = loader_cls(path)
        documents.extend(loader.load())

    return documents
