# LLM Fundamentals Explained

A fourteen-part study guide for building with large language models — from how a model reads text, through prompt engineering, provider APIs and LangChain, to structured outputs, conversation memory, streaming, embeddings and RAG, evaluation, tool calling, and agents.

Every part follows the same three-beat pattern: **what it is → how it works → why it matters**, with runnable Python examples and a quick-reference cheat sheet at the end of most parts.

> **How to read this:** start at Part 1 and work forward, or jump straight to the part you need — every file is self-contained. Source: a Q&A walkthrough (Nov 2026); the code samples use the model IDs from that session, so swap in current ones when you run them.

---

## The parts

### [1. LLM Fundamentals](01-llm-fundamentals.md)

How the model itself works: tokens, context windows, temperature, sampling, and the failure modes to expect.

*Tokens · Context windows · Temperature · Top-p / nucleus sampling · Hallucinations · Limitations*

### [2. Prompt Engineering](02-prompt-engineering.md)

Who says what: system prompts set the rules, user prompts set the task, and few-shot examples show the pattern.

*System prompts · User prompts · Few-shot prompting*

### [3. LLM APIs and SDKs](03-llm-apis-and-sdks.md)

The plumbing between your code and a provider: requests, messages, responses, keys, and rate limits.

*LLM APIs · SDKs · Requests · Messages · Responses · API keys · .env files · Rate limits & errors*

### [4. LangChain Basics](04-langchain-basics.md)

One interface for every provider: install, import, and swap models with `init_chat_model`.

*Overview · Installation · langchain vs langchain-core · init_chat_model · OpenAI · Gemini · Anthropic · Groq*

### [5. Messages and Prompt Templates](05-messages-and-prompt-templates.md)

The three message roles, string templates, chat templates, variables, and reusable prompt objects.

*Message types · PromptTemplate · ChatPromptTemplate · Variables in prompts · MessagesPlaceholder · Dynamic prompts · Reusing templates*

### [6. Output Handling](06-output-handling.md)

What actually comes back from `.invoke()` and how to read it safely.

*Raw responses · Extracting content · Metadata, usage & errors · Invalid or unexpected output*

### [7. Structured Outputs](07-structured-outputs.md)

Turning free text into typed Python objects with JSON schemas, Pydantic, and `with_structured_output()`.

*Why structure matters · JSON responses · Schemas · Pydantic · with_structured_output() · Nested schemas · Validation errors*

### [8. Conversation History](08-conversation-history.md)

Giving a stateless model memory — and stopping that memory from overflowing the context window.

*The stateless problem · Message history · ChatMessageHistory · RunnableWithMessageHistory · Context overflow · Trimming strategies · Persistence*

### [9. Streaming and Async](09-streaming-and-async.md)

Showing tokens as they are produced, and running calls concurrently with async and batching.

*Why streaming matters · Token-by-token streaming · .stream() · .astream() · FastAPI endpoints · Async & batching · Error handling*

### [10. Embeddings and Vector Stores](10-embeddings-and-vector-stores.md)

Turning meaning into numbers, then searching it: similarity metrics, vector databases, retrievers, and the full document pipeline.

*Embeddings · Semantic similarity · Generating embeddings · Vector stores · Document retrieval pipeline · Retrievers & search strategies · Limitations & best practices*

### [11. RAG (Retrieval-Augmented Generation)](11-rag-retrievalaugmented-generation.md)

Load, split, index, retrieve, generate — grounding answers in your own documents, end to end.

*The RAG concept · Document loading · Text splitting · Indexing · Retrieval · Generation · End-to-end pipeline · Failure modes · Advanced patterns*

### [12. Evaluation and Reliability](12-evaluation-and-reliability.md)

Testing probabilistic software: golden datasets, semantic and LLM-judge scoring, groundedness, retries, and guardrails.

*Why evaluate · Quality dimensions · Golden datasets · Deterministic / semantic / judge tests · RAG evaluation · Failures & retries · Reducing hallucinations · Production reliability*

### [13. Tool Calling](13-tool-calling.md)

Letting the model request actions: function schemas, generated arguments, tool results, and the execution loop.

*Tool calling · Function schemas · @tool + Pydantic · Arguments as untrusted input · Tool results · The full loop · Parallel tool calls · Errors & timeouts · Security*

### [14. Agent Fundamentals](14-agent-fundamentals.md)

LLM + tools + loop: the agent cycle, tool selection, stopping conditions, state, and production patterns.

*What an agent is · The agent loop · Tool selection · Stopping conditions · State & memory · Manual loop example · LangGraph & prebuilt agents · Failure modes · Evaluation*

---

## At a glance

| # | Part | File |
|---|---|---|
| 1 | LLM Fundamentals | [`01-llm-fundamentals.md`](01-llm-fundamentals.md) |
| 2 | Prompt Engineering | [`02-prompt-engineering.md`](02-prompt-engineering.md) |
| 3 | LLM APIs and SDKs | [`03-llm-apis-and-sdks.md`](03-llm-apis-and-sdks.md) |
| 4 | LangChain Basics | [`04-langchain-basics.md`](04-langchain-basics.md) |
| 5 | Messages and Prompt Templates | [`05-messages-and-prompt-templates.md`](05-messages-and-prompt-templates.md) |
| 6 | Output Handling | [`06-output-handling.md`](06-output-handling.md) |
| 7 | Structured Outputs | [`07-structured-outputs.md`](07-structured-outputs.md) |
| 8 | Conversation History | [`08-conversation-history.md`](08-conversation-history.md) |
| 9 | Streaming and Async | [`09-streaming-and-async.md`](09-streaming-and-async.md) |
| 10 | Embeddings and Vector Stores | [`10-embeddings-and-vector-stores.md`](10-embeddings-and-vector-stores.md) |
| 11 | RAG (Retrieval-Augmented Generation) | [`11-rag-retrievalaugmented-generation.md`](11-rag-retrievalaugmented-generation.md) |
| 12 | Evaluation and Reliability | [`12-evaluation-and-reliability.md`](12-evaluation-and-reliability.md) |
| 13 | Tool Calling | [`13-tool-calling.md`](13-tool-calling.md) |
| 14 | Agent Fundamentals | [`14-agent-fundamentals.md`](14-agent-fundamentals.md) |

## Suggested reading paths

| If you want to… | Read |
|---|---|
| Understand the model itself | Parts 1–2 |
| Call an API from Python for the first time | Part 3 |
| Build with LangChain | Parts 4–9 |
| Get reliable, machine-readable output | Part 7 |
| Ship a chatbot that remembers and streams | Parts 8–9 |
| Ground answers in your own documents | Parts 10–11 |
| Test and harden an LLM app | Part 12 |
| Let the model take actions | Part 13 |
| Build an agent | Part 14 |
