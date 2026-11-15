# 11. RAG (Retrieval-Augmented Generation)

[Home](README.md) · Part 11 of 14

> Load, split, index, retrieve, generate — grounding answers in your own documents, end to end.

**In this part:** The RAG concept · Document loading · Text splitting · Indexing · Retrieval · Generation · End-to-end pipeline · Failure modes · Advanced patterns

---

## 1. The RAG Concept (Why Not Just Ask the LLM?)
**What it is:** RAG (Retrieval-Augmented Generation) is a pattern where you first **retrieve** relevant information from your own documents, then **augment** the LLM's prompt with that information, and finally let the LLM **generate** an answer grounded in that evidence.
- **The Problem RAG Solves:**
    - LLMs have a knowledge cutoff and don't know your private data.
    - Stuffing an entire knowledge base into the context window is expensive, slow, and hits the "lost in the middle" problem.
    - Fine-tuning teaches *style*, not *facts* — it's the wrong tool for injecting fresh knowledge.
- **The RAG Flow:**

```text
OFFLINE (Indexing):
Documents → Load → Split → Embed → Store in Vector DB

ONLINE (Query time):
Question → Embed → Retrieve top-k chunks → Build Prompt → LLM → Answer
```

- **Why it matters:** RAG turns a general-purpose LLM into a system that can answer questions about *your* data — with citations, freshness, and access control — without retraining the model.

## 2. Document Loading (Getting Data In)
**What it is:** Converting raw files (PDF, HTML, Markdown, CSV, Notion, etc.) into a standardized `Document` object that LangChain can process.
- **The `Document` structure:**

```python
from langchain_core.documents import Document

doc = Document(
    page_content="The actual text content of the document...",
    metadata={
        "source": "handbook.pdf",
        "page": 5,
        "author": "HR Team",
        "created_at": "2026-01-15"
    }
)
```

- **Common Loaders:**

```bash
pip install langchain-community pypdf unstructured
```

```python
# PDF
from langchain_community.document_loaders import PyPDFLoader
docs = PyPDFLoader("handbook.pdf").load()

# Web page
from langchain_community.document_loaders import WebBaseLoader
docs = WebBaseLoader("https://example.com/docs").load()

# Directory of text files
from langchain_community.document_loaders import DirectoryLoader, TextLoader
docs = DirectoryLoader("./docs/", glob="**/*.md", loader_cls=TextLoader).load()

# CSV
from langchain_community.document_loaders import CSVLoader
docs = CSVLoader("data.csv").load()
```

- **Why it matters:** Loaders normalize messy real-world formats into a consistent structure. Crucially, they preserve **metadata** (page numbers, sources) which enables citations and filtering later.

## 3. Text Splitting (Chunking Strategy)
**What it is:** Breaking long documents into smaller, semantically coherent chunks that can be embedded and retrieved precisely.
- **Why you must split:**
    - Embedding a 50-page document as one vector averages all its meanings — retrieval becomes vague.
    - Small chunks produce sharp, specific vectors that match specific questions.
    - Retrieved chunks must fit in the context window alongside the question.
- **The Standard Splitter:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=150,    # Overlap to preserve context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]  # Try to split at natural boundaries
)

chunks = splitter.split_documents(docs)
print(f"Split {len(docs)} documents into {len(chunks)} chunks")
```

- **How `RecursiveCharacterTextSplitter` works:** It tries to split at paragraph breaks first (`\n\n`). If a chunk is still too big, it falls back to line breaks, then sentences, then words. This keeps related text together.
- **Strategy Comparison:**

| Strategy | Best For | Trade-off |
|---|---|---|
| Fixed size | Simple, fast | Cuts mid-sentence |
| Recursive character | General purpose (default choice) | Good balance |
| Markdown/HTML header-aware | Structured docs | Preserves section hierarchy |
| Semantic chunking | Highest quality | Slower, costs embedding calls |
| Code-aware (by language) | Source code | Splits at functions/classes |

- **Code-aware example:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100
)
```

- **Why it matters:** Chunking is the single highest-leverage decision in RAG. Bad chunks → bad retrieval → bad answers, no matter how good your LLM is.

## 4. Indexing (Embeddings + Vector Store)
**What it is:** Converting chunks into embedding vectors and storing them in a searchable index. This is the **offline** step you do once (and re-do when documents change).
- **How it works:**

```python
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# One call: embed all chunks + store them
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./rag_index",
    collection_name="company_docs"
)
```

- **Loading an existing index (skip re-embedding):**

```python
# On subsequent runs, just load — don't rebuild
vectorstore = Chroma(
    persist_directory="./rag_index",
    embedding_function=embeddings,
    collection_name="company_docs"
)
```

- **Incremental updates (avoid re-embedding everything):**

```python
# Add new documents to existing index
vectorstore.add_documents(new_chunks)

# Delete by ID
vectorstore.delete(ids=["doc_id_1", "doc_id_2"])
```

- **Why it matters:** Embedding costs money and time. Building the index once and persisting it means your app starts instantly and you only pay to embed new/changed documents.

## 5. Retrieval (Finding the Right Chunks)
**What it is:** At query time, embed the user's question and find the most semantically similar chunks in the index.
- **Basic retrieval:**

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

relevant_docs = retriever.invoke("What is the parental leave policy?")

for doc in relevant_docs:
    print(f"[{doc.metadata['source']} p.{doc.metadata.get('page')}]")
    print(doc.page_content[:200])
    print("---")
```

- **Advanced Retrieval Strategies:**

| Strategy | What it does | When to use |
|---|---|---|
| **Similarity (top-k)** | Return k closest vectors | Default starting point |
| **MMR** | Balances relevance + diversity | Avoid k near-duplicate chunks |
| **Score threshold** | Only return matches above a relevance floor | Filter out weak matches |
| **Metadata filter** | Restrict by source, date, permissions | Multi-tenant, access control |
| **Hybrid search** | Vector + keyword (BM25) | Exact IDs, names, codes |
| **Reranking** | Second-pass model reorders results | Noisy top-k results |

- **MMR (diversity):**

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
)
```

- **Score threshold:**

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 4}
)
```

- **Metadata filtering (access control):**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"department": "engineering"}
    }
)
```

- **Why it matters:** Retrieval is where most RAG systems fail. If the right chunk isn't retrieved, the LLM cannot possibly answer correctly. Invest here first.

## 6. Generation (Answering with Context)
**What it is:** Feeding the retrieved chunks into a prompt and having the LLM synthesize a grounded answer.
- **The RAG Prompt Pattern:**

```python
from langchain_core.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the "
     "provided context. If the context doesn't contain the answer, say "
     "'I don't have enough information to answer that.'\n\n"
     "Cite sources using [source] notation.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])
```

- **Building the full RAG chain (LCEL):**

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def format_docs(docs):
    return "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}"
        for doc in docs
    )

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the parental leave policy?")
print(answer)
```

- **Getting sources back (for transparency):**

```python
from langchain_core.runnables import RunnableParallel

rag_chain_with_sources = RunnableParallel(
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
)

# Retrieve first, inspect, then generate
docs = retriever.invoke("What is the parental leave policy?")
print("Retrieved sources:", [d.metadata["source"] for d in docs])

answer = rag_chain.invoke("What is the parental leave policy?")
```

- **Why it matters:** The generation step is where you enforce grounding ("answer only from context"), handle missing information gracefully, and produce citations. Without these instructions, the LLM will happily fill gaps with hallucinations.

## 7. End-to-End RAG Pipeline (Complete Working Example)
**What it is:** All five stages wired together in a single script.
- **Full Implementation:**

```python
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# ─── STAGE 1: LOAD ───
docs = PyPDFLoader("handbook.pdf").load()

# ─── STAGE 2: SPLIT ───
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
)
chunks = splitter.split_documents(docs)

# ─── STAGE 3: INDEX ───
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./rag_index"
)

# ─── STAGE 4: RETRIEVE ───
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ─── STAGE 5: GENERATE ───
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the context below. If unsure, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ─── QUERY ───
answer = rag_chain.invoke("How many vacation days do employees get?")
print(answer)
```

- **Why it matters:** This is the minimal viable RAG system. Every production RAG app is an elaboration of these five stages.

## 8. Common Failure Modes and Fixes
**What it is:** RAG systems fail in predictable ways. Knowing the patterns saves debugging time.
- **Diagnostic Table:**

| Symptom | Likely Cause | Fix |
|---|---|---|
| "I don't know" to answerable questions | Retrieval missing the chunk | Increase k, better chunking, hybrid search |
| Confident but wrong answers | Hallucination beyond context | Stricter prompt, lower temperature |
| Right topic, wrong detail | Chunk too large/small | Tune chunk_size |
| Answers cite wrong page | Metadata lost during splitting | Verify metadata propagates |
| Duplicate content in context | No diversity in retrieval | Use MMR |
| Can't find exact product codes | Pure vector search weakness | Add keyword/BM25 hybrid search |
| Slow responses | Too many chunks, large context | Reduce k, rerank, compress context |
| Answers change randomly | High temperature | Set temperature=0 |

- **Debugging Tip — Always inspect retrieval first:**

```python
# Before blaming the LLM, check what was actually retrieved
docs = retriever.invoke("your question here")
for i, doc in enumerate(docs):
    print(f"\n=== Chunk {i+1} (score/source: {doc.metadata}) ===")
    print(doc.page_content)
```

- **Why it matters:** 80% of RAG problems are retrieval problems, not generation problems. If the wrong context arrives, no prompt engineering can save you.

## 9. Advanced RAG Patterns (Beyond the Basics)
**What it is:** Techniques to improve RAG quality when the basic pipeline isn't enough.
- **Key Patterns:**

| Pattern | What it does |
|---|---|
| **Query rewriting** | LLM rephrases the question for better retrieval |
| **Multi-query** | Generate 3 question variants, retrieve for each, merge results |
| **HyDE** | Generate a hypothetical answer, embed *that* instead of the question |
| **Reranking** | Cross-encoder rescores top-20 → returns best 4 |
| **Parent document retrieval** | Embed small chunks, return the larger parent section |
| **Contextual compression** | LLM extracts only the relevant sentences from retrieved chunks |
| **Hybrid search** | Combine vector similarity + keyword BM25 scores |
| **Self-RAG / CRAG** | Model grades its own retrieval and retries if poor |

- **Reranking example:**

```python
# Retrieve broadly, then rerank for precision
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20})
)

compressed_docs = compression_retriever.invoke("What is the refund policy?")
```

- **Why it matters:** Basic RAG gets you 70% of the way. These patterns close the gap for production-quality accuracy, especially on complex or ambiguous questions.

## Quick Reference Cheat Sheet

| Stage | Key Tool | Purpose |
|---|---|---|
| **Load** | `PyPDFLoader`, `WebBaseLoader`, `DirectoryLoader` | Files → `Document` objects |
| **Split** | `RecursiveCharacterTextSplitter` | Long docs → precise chunks |
| **Index** | `OpenAIEmbeddings` + `Chroma.from_documents()` | Chunks → searchable vectors |
| **Retrieve** | `vectorstore.as_retriever(k=4)` | Question → relevant chunks |
| **Generate** | `prompt \| llm \| StrOutputParser()` | Context + question → answer |
| **Diversity** | `search_type="mmr"` | Avoid duplicate chunks |
| **Filtering** | `search_kwargs={"filter": {...}}` | Access control, scoping |
| **Precision** | `ContextualCompressionRetriever` | Rerank/compress results |
| **Grounding** | "Answer only from context" in system prompt | Reduce hallucination |
| **Debugging** | Inspect `retriever.invoke()` output | Verify retrieval before generation |

---

← [10. Embeddings and Vector Stores](10-embeddings-and-vector-stores.md) · [Contents](README.md) · [12. Evaluation and Reliability](12-evaluation-and-reliability.md) →
