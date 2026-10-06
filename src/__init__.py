# Makes src/ a Python package so `from src.document_loader import ...` works.
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)