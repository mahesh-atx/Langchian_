# 10. Embeddings and Vector Stores

[Home](README.md) · Part 10 of 14

> Turning meaning into numbers, then searching it: similarity metrics, vector databases, retrievers, and the full document pipeline.

**In this part:** Embeddings · Semantic similarity · Generating embeddings · Vector stores · Document retrieval pipeline · Retrievers & search strategies · Limitations & best practices

---

## 1. Embeddings (Meaning as Numbers)
**What it is:** An embedding is a numeric vector — a list of floats — that represents the *semantic meaning* of text, images, audio, or code. Similar meanings produce vectors that are close together in vector space.
- **How it works:** An embedding model converts text into a fixed-length array, e.g. 1536 or 3072 numbers.

```text
"I love apples"  → [0.021, -0.183, 0.442, ..., 0.077]
"Apples are my favorite fruit" → [0.019, -0.176, 0.451, ..., 0.081]
"Quarterly tax filing deadline" → [-0.312, 0.044, -0.019, ..., 0.265]
```

- **Key idea:** The first two sentences share few exact words but have similar meaning, so their vectors are close. The third is unrelated, so its vector is far away.
- **Why it matters:** Embeddings let you search by *meaning* instead of exact keywords. This is the foundation of semantic search, recommendations, clustering, deduplication, and RAG.

## 2. Semantic Similarity (Closeness = Relatedness)
**What it is:** Semantic similarity measures how related two pieces of text are by comparing their embedding vectors.
- **Common metrics:**

| Metric | Intuition | Notes |
|---|---|---|
| Cosine similarity | Angle between vectors | Most common for text embeddings |
| Dot product | Alignment + magnitude | Often used with normalized vectors |
| Euclidean distance | Straight-line distance | Lower distance = more similar |

- **Example:**

```python
import numpy as np

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Higher score = more semantically similar
score = cosine_similarity(vector_1, vector_2)
```

- **Important nuance:** Similarity is not truth. Two sentences can be semantically similar but factually contradictory:

```text
"The meeting is on Monday."
"The meeting is not on Monday."
```

Embeddings may still place them close because they discuss the same topic.
- **Why it matters:** Vector search retrieves *related* content, not necessarily *correct* or *sufficient* content. You still need good chunking, filtering, and sometimes reranking.

## 3. Generating Embeddings in LangChain
**What it is:** LangChain provides embedding classes that convert documents and queries into vectors.
- **Installation:**

```bash
pip install langchain langchain-openai langchain-chroma langchain-text-splitters chromadb python-dotenv
```

- **Example:**

```python
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # Loads OPENAI_API_KEY from .env

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Embed multiple documents
vectors = embeddings.embed_documents([
    "LangChain helps developers build LLM applications.",
    "Vector stores enable semantic search over documents."
])

# Embed a search query
query_vector = embeddings.embed_query("How do I search documents by meaning?")

print(len(vectors))        # Number of embedded documents
print(len(vectors[0]))     # Vector dimension, e.g. 1536
```

- **Rule:** Use the **same embedding model** for documents and queries. If you change embedding models, you usually must re-embed your entire corpus.
- **Why it matters:** `embed_documents()` and `embed_query()` are the two core operations. Documents are embedded once and stored; queries are embedded at search time.

## 4. Vector Stores (Searchable Semantic Memory)
**What it is:** A vector store is a database or index that stores embeddings alongside the original text and metadata, then retrieves the closest matches for a query vector.
- **How it works:**
    1. Store text chunks.
    2. Convert each chunk into an embedding.
    3. Save the vector + original text + metadata.
    4. At query time, embed the query.
    5. Find the nearest vectors.
    6. Return the corresponding text chunks.
- **Common options:**

| Store | Type | Good for |
|---|---|---|
| FAISS | Local vector index library | Fast local experiments, offline search |
| Chroma | Embedded/local vector DB | Prototypes, small apps, persistent local storage |
| pgvector | PostgreSQL extension | Apps already using Postgres |
| Qdrant / Weaviate | Vector databases | Filtering, production search |
| Pinecone | Managed vector DB | Serverless/managed production workloads |

- **LangChain example with Chroma:**

```python
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(
        page_content="LangChain provides tools for building applications with LLMs.",
        metadata={"source": "intro.md", "topic": "langchain"}
    ),
    Document(
        page_content="Vector stores index embeddings for semantic retrieval.",
        metadata={"source": "vectors.md", "topic": "retrieval"}
    ),
    Document(
        page_content="Prompt templates help structure model inputs.",
        metadata={"source": "prompts.md", "topic": "prompting"}
    )
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

results = vectorstore.similarity_search("How can I build LLM apps?", k=2)

for doc in results:
    print(doc.page_content)
    print(doc.metadata)
```

- **Why it matters:** The LLM itself does not “remember” your documents. The vector store acts as external memory that can be searched and injected into the prompt.

## 5. Document Retrieval Pipeline (From Files to Answers)
**What it is:** Document retrieval is the process of finding the most relevant pieces of your private or external knowledge base before asking the LLM to answer.
- **Standard pipeline:**

```text
Load documents
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store vectors + metadata
    ↓
Embed user query
    ↓
Retrieve top-k relevant chunks
    ↓
Put chunks into prompt
    ↓
LLM generates grounded answer
```

- **Why chunking is needed:** A whole book or PDF is too large to embed as one useful vector. Small chunks produce more precise retrieval.
- **Chunking example:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

raw_docs = [
    Document(
        page_content="Long document text goes here..." * 200,
        metadata={"source": "handbook.pdf"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(raw_docs)
```

- **Typical starting point:**
    - `chunk_size`: 500–1500 characters or roughly 200–500 tokens
    - `chunk_overlap`: 10–20% of chunk size
- **Why it matters:** Retrieval quality depends heavily on chunk quality. Bad chunks produce bad context, which produces bad answers.

## 6. Retrievers and Search Strategies
**What it is:** A retriever is a LangChain interface that takes a query string and returns relevant documents.
- **Basic retriever:**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

docs = retriever.invoke("What is a vector store?")
```

- **Search with scores:**

```python
results = vectorstore.similarity_search_with_score(
    "semantic document search",
    k=3
)

for doc, score in results:
    print(score, doc.page_content)
```

**Warning:** Score meaning differs by vector store. Some return distances where lower is better; others return similarity where higher is better.
- **MMR search for diversity:**

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20
    }
)
```

MMR — Maximal Marginal Relevance — tries to return relevant results that are not all near-duplicates.
- **Metadata filtering:**

```python
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": "handbook.pdf"}
    }
)
```

Filter syntax varies by vector store, but the idea is the same: restrict search by source, user, date, department, permissions, etc.
- **Why it matters:** Plain top-k similarity is often not enough. Production retrieval usually needs filtering, diversity, reranking, or hybrid keyword + vector search.

## 7. RAG Example (Retrieval-Augmented Generation)
**What it is:** RAG means retrieving relevant documents and giving them to the LLM as context before it answers.
- **How it works:** The model answers from retrieved evidence instead of relying only on its training data.
- **Complete example:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

docs = [
    Document(
        page_content="Refunds are available within 30 days of purchase.",
        metadata={"source": "policy.md", "section": "refunds"}
    ),
    Document(
        page_content="Customers can contact support via email or live chat.",
        metadata={"source": "policy.md", "section": "support"}
    )
]

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the provided context. "
     "If the answer is not in the context, say you don't know.\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[{i+1}] {doc.page_content} "
        f"(source: {doc.metadata.get('source', 'unknown')})"
        for i, doc in enumerate(docs)
    )

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

answer = chain.invoke("What is the refund policy?")
print(answer)
```

- **Why it matters:** RAG reduces hallucinations by grounding answers in retrieved documents. It does **not** eliminate hallucinations — the model can still misread context or answer beyond it.

## 8. Limitations and Best Practices
**What it is:** Embeddings and vector search are powerful, but they are not magic databases of truth.
- **Common failure modes:**

| Problem | Example | Fix |
|---|---|---|
| Similar ≠ relevant | Retrieves related but unhelpful chunks | Better chunking, reranking |
| Exact match weakness | Part numbers, IDs, legal citations | Hybrid keyword + vector search |
| Negation confusion | “approved” vs “not approved” | Reranking, stronger prompts |
| Stale index | Old policy documents retrieved | Re-index on updates |
| Wrong embedding model | Query and docs embedded differently | Use one consistent model |
| Permission leaks | User retrieves restricted docs | Metadata/access filters |
| Too much context | Top-k returns noise | Lower k, rerank, compress |

- **Production tips:**
    - Store metadata: `source`, `page`, `section`, `timestamp`, `user_id`, `permissions`.
    - Log retrieved chunks and scores before blaming the LLM.
    - Start with `k=3–5`, then tune.
    - Use hybrid search for exact terms, codes, names, and IDs.
    - Use a reranker when top-k results are noisy.
    - Re-embed everything when changing embedding models.
    - Evaluate retrieval separately from generation.
- **Why it matters:** In RAG systems, retrieval failures are often the real cause of bad answers. If the wrong context is retrieved, even a perfect LLM will produce a wrong or unsupported response.

## Quick Reference Cheat Sheet

| Concept | Tool / Method | Purpose |
|---|---|---|
| Embed documents | `embeddings.embed_documents(texts)` | Convert stored text to vectors |
| Embed query | `embeddings.embed_query(query)` | Convert search question to vector |
| Create vector store | `Chroma.from_documents(...)` | Store docs + embeddings |
| Similarity search | `vectorstore.similarity_search(query, k=4)` | Retrieve closest chunks |
| Search with scores | `similarity_search_with_score(...)` | Inspect match quality |
| Retriever | `vectorstore.as_retriever()` | Standard retrieval interface |
| Diverse retrieval | `search_type="mmr"` | Reduce duplicate results |
| Metadata filter | `search_kwargs={"filter": {...}}` | Restrict by source/user/date |
| Chunking | `RecursiveCharacterTextSplitter` | Split long docs into searchable pieces |
| RAG chain | `retriever → prompt → llm` | Ground answers in documents |

---

← [9. Streaming and Async](09-streaming-and-async.md) · [Contents](README.md) · [11. RAG (Retrieval-Augmented Generation)](11-rag-retrievalaugmented-generation.md) →
