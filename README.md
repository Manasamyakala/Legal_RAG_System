# RAG from Scratch 

A beginner-friendly implementation of Retrieval-Augmented Generation (RAG) built step-by-step using LangChain, FAISS, and HuggingFace embeddings. Every file is heavily commented to explain *why* each piece exists, not just what it does.

## See It Working

Two real runs, no OpenAI key needed — this uses a small model running locally on this machine via [Ollama](#using-ollama-no-api-key-needed).

The most important test for any RAG system: does it actually answer from your documents, and does it admit when it doesn't know rather than making something up?

And a real question answered from the actual PDF content that ships in `Learnings/`:

**LLM (Large Language Model).** This is the "AI" part — programs like GPT-4, Claude, or Llama. An LLM has read an enormous amount of text (books, websites, articles) and learned to predict what word comes next well enough to hold a conversation, answer questions, and write like a person. Think of it as an extremely well-read assistant with one catch: it has never seen *your* documents, and everything it knows stops at whatever date it was trained.

**Embedding.** A way of turning a piece of text into a list of numbers that captures its *meaning*, not just its exact words. Two sentences that mean similar things end up as very similar lists of numbers — even if they don't share a single word. Think of it like giving every sentence GPS coordinates on a map of *ideas*: "dog" and "puppy" land close together; "dog" and "stock market" land far apart.

**Vector.** That list of numbers is called a vector. In this project, every chunk of text becomes a vector of 384 numbers. You never see these numbers directly — the computer uses them to measure how close two pieces of text are in meaning.

**Chunk / Chunking.** Cutting a long document into smaller pieces — a paragraph or two at a time — before turning each piece into a vector. Why bother? Comparing "meaning" works best on one focused idea at a time, not an entire 50-page manual lumped into a single vector.

**FAISS.** A free tool (built by Meta) that can hold millions of these vectors and instantly find the ones closest in meaning to a new question. Think of it as a librarian with a photographic memory for exactly where every idea "lives" on the meaning-map, who can point you to the closest match in a fraction of a second — instead of you flipping through every page yourself.

**LangChain.** A toolkit of ready-made building blocks for wiring all of the above together — reading files, chunking text, calling an embedding model, talking to FAISS, and finally sending everything to an LLM — so this project doesn't have to write all of that plumbing from scratch.

**RAG (Retrieval-Augmented Generation).** The idea that ties everything together. Imagine asking a very well-read friend a question, but instead of trusting their memory (which might be outdated, or just wrong), you hand them the exact relevant pages from your own notes and say: "Answer using only this." That's RAG — **Retrieval** (find the relevant pages) plus **Augmented Generation** (let the LLM write an answer using those pages, not its own memory).

### What Actually Happens When You Ask a Question — In Plain English

1. **Your documents get cut into small pieces.** A 10-page PDF might become 40 small chunks, a few paragraphs each.
2. **Each chunk gets a "meaning fingerprint."** Every chunk is converted into a vector — a list of numbers standing in for what that chunk is *about*.
3. **All those fingerprints get filed away.** FAISS stores every chunk's vector, like a card catalog sorted by meaning instead of alphabetically.
4. **You ask a question.** Your question gets the exact same "meaning fingerprint" treatment — turned into a vector the same way.
5. **The librarian finds the closest matches.** FAISS compares your question's vector against every stored chunk and hands back the 3 (by default) closest ones.
6. **Those matching chunks — and only those — go to the LLM**, along with an explicit instruction: "answer using only this information, and say 'I don't know' if it isn't here."
7. **The LLM writes a grounded answer**, and this project also prints which files the answer came from, so you can double-check it yourself.

No step here requires you to write code to *understand* it — the code in `src/` is just each of these seven steps as one small Python file, heavily commented so you can match what you just read above to exactly where it happens.

## What is RAG and Why Does It Matter?

**The problem with plain LLMs:** Large Language Models like GPT-4 are trained on data up to a certain cutoff date, and they have no knowledge of your private documents — your company's policy manuals, your research papers, your product documentation. If you ask GPT-4 "What is the refund policy in our internal handbook?", it simply doesn't know.

**What RAG does:** RAG (Retrieval-Augmented Generation) solves this by giving the LLM access to your documents at query time. Instead of retraining the model (expensive, slow), you store your documents in a searchable vector database. When a user asks a question, you retrieve the most relevant passages and include them in the LLM's prompt. The LLM reads those passages and answers based on your documents.

**Why it matters:** RAG is currently the dominant architecture for production AI Q&A systems. It's cost-effective (no retraining), updatable (just add documents to the database), and auditable (you can see exactly which document chunks informed each answer). Understanding RAG from scratch gives you the foundation to build everything from customer support bots to internal knowledge assistants.

## Step-by-Step Setup

**1. Create a virtual environment**

```
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```
pip install -r requirements.txt
```

⏱️ First install may take a few minutes. `faiss-cpu` and `sentence-transformers` are the largest packages.

**3. Configure your API key**

```
cp .env.example .env
```

Open `.env` and replace `your_openai_api_key_here` with your actual key from platform.openai.com.

```
OPENAI_API_KEY=sk-...your-key-here...
```

💡 No OpenAI account? Use a local model with Ollama — see [Using Ollama](#using-ollama-no-api-key-needed) below.

**4. Add your documents**

Drop any `.pdf`, `.txt`, or `.docx` files into:

```
Learnings/
```

This repo already ships with 7 real PDFs (AI/ML learning material) plus 4 fictional company-handbook text files — refund, leave, and security policy, and a test file for the hallucination check below — so `python main.py` works immediately with no setup. The more documents you add, the more the system can answer.

**5. Run it!**

```
# Interactive mode — asks questions in a loop
python main.py

# Single question mode
python main.py --question "What are the main topics in these documents?"

# Debug mode — shows retrieved chunks and full LLM prompt
python main.py --debug --question "What is the refund policy?"
```

## How to Add Your Own Documents

Just drop files into `Learnings/`. The loader automatically detects file types:

| File type | Support | Notes |
|---|---|---|
| `.pdf` | ✅ | Each page becomes a separate Document |
| `.txt` | ✅ | Entire file is one Document |
| `.docx` | ✅ | Entire file is one Document |
| `.csv` | ❌ | Not supported (yet) |

After adding new documents, delete the cached FAISS index so it gets rebuilt:

```
rm -rf faiss_index/
python main.py
```

## How to Verify the LLM Uses Your Documents

This is the most important test for any RAG system — make sure it's actually reading your documents and not falling back on general knowledge.

**Step 1:** `Learnings/zorbax_protocol.txt` already ships in this repo with a very specific, obscure fact:

```
The Zorbax Protocol was established in 2019 by Dr. Eleanor Voss.
The protocol requires three phases: initialization, calibration, and review.
```

**Step 2:** Ask the system about it:

```
python main.py --question "Who established the Zorbax Protocol?"
```

Expected good result:

```
Answer: Dr. Eleanor Voss established the Zorbax Protocol in 2019.
Sources: Learnings/zorbax_protocol.txt
```

**Step 3:** Ask about something NOT in any document:

```
python main.py --question "What is the capital of Australia?"
```

Expected good result:

```
Answer: I don't know based on the provided documents.
```

If the second answer returns "Canberra" (from general knowledge), the system is hallucinating — check that your prompt template in `src/generator.py` is being applied correctly.

## Using Ollama (No API Key Needed)

Ollama lets you run LLMs locally for free.

```
# 1. Install Ollama: https://ollama.com
# 2. Pull a model
ollama pull llama3      # ~4GB download
ollama pull mistral     # ~4GB download, often faster

# 3. Run with Ollama
python main.py --model ollama/llama3
python main.py --model ollama/mistral --question "Summarize the documents"
```

## Beginner Tips

**What happens if chunk_size is too large or too small?**

| Setting | Effect |
|---|---|
| chunk_size too large (e.g., 2000) | Fewer chunks, less precise retrieval. The LLM receives a lot of text, most of which may be irrelevant to the question. |
| chunk_size too small (e.g., 50) | Thousands of tiny chunks. Each chunk lacks context — a sentence like "See the above section" becomes meaningless on its own. |
| Sweet spot (300–800 chars) | Roughly 1–2 paragraphs. Enough context to be meaningful, small enough to be precise. |

**Why cosine similarity beats keyword search**

Traditional search (e.g., grep, SQL LIKE) requires exact word matches. Search for "car" and you won't find documents that say "automobile" or "vehicle".

Semantic search (cosine similarity over embeddings) understands meaning:
- "car", "automobile", "vehicle", "sedan" → all have very similar embeddings
- You can ask "What's the fastest way to travel?" and find chunks about "high-speed rail" or "airplane travel" — no exact keyword overlap needed

**What does k mean in top-k retrieval?**

`k` is the number of document chunks retrieved per question.
- `k=1`: Only the single best match. Very precise but may miss relevant context.
- `k=3` (default): A good balance. Captures the primary answer + nearby supporting text.
- `k=10`: Comprehensive but may include loosely related chunks that dilute the LLM's focus.

Use `--k 5` on the command line to experiment. If the LLM keeps saying "I don't know" on questions you know are in the docs, try increasing k.

## Troubleshooting

**`OPENAI_API_KEY` is not set**

```
cp .env.example .env
# edit .env and add your key
```

**No documents were loaded**

Make sure you have files in `Learnings/`. Only `.pdf`, `.txt`, and `.docx` are supported.

**`FileNotFoundError: Learnings does not exist`**

```
mkdir -p Learnings
# then add your files
```

**`Error: Connection refused` (Ollama)**

Make sure Ollama is running: `ollama serve`

**Model not found (Ollama)**

Pull the model first: `ollama pull llama3`

**Answers seem wrong or generic**

- Run with `--debug` to see which chunks are being retrieved
- Check the sources printed after each answer — are they the right files?
- Try deleting `faiss_index/` and rebuilding — you may have stale embeddings
- Try increasing `--k` to retrieve more context

**`pip install` fails on `faiss-cpu`**

On some systems you may need to install build tools:

```
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```
