# 8. Conversation History

[Home](README.md) · Part 8 of 14

> Giving a stateless model memory — and stopping that memory from overflowing the context window.

**In this part:** The stateless problem · Message history · ChatMessageHistory · RunnableWithMessageHistory · Context overflow · Trimming strategies · Persistence · Best practices

---

## 1. The Stateless Problem (Why History Matters)
**What it is:** LLMs are inherently **stateless**. Every API call is completely independent. The model has zero memory of your previous conversations. If you send "What is my name?" in call #2, the model has no idea you told it your name in call #1.
- **How it works:** To simulate memory, *your application* must store the entire conversation and re-send it with every new request.
- **The Pattern:**

```
Call 1: [User: "My name is Alice."]           → AI: "Hello Alice!"
Call 2: [User: "My name is Alice."]           → AI: "Hello Alice!"
        [AI: "Hello Alice!"]
        [User: "What is my name?"]            → AI: "Your name is Alice."
```

- **Why it matters:** Without managing history, every interaction starts from scratch. Conversation history is what transforms a "single-query engine" into a "chatbot."

## 2. Message History (The Conversation Log)
**What it is:** An ordered list of all messages exchanged between the user and the AI during a session. This is the raw data that gets re-sent to the LLM on every turn.
- **How it works:** Each message is stored with its `role` (system, human/user, ai/assistant) and `content`. The list grows with every exchange.
- **Basic Example (Manual):**

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Start with system instructions
history = [
    SystemMessage(content="You are a helpful travel assistant.")
]

# Turn 1
history.append(HumanMessage(content="I want to visit Japan."))
response = llm.invoke(history)
history.append(AIMessage(content=response.content))

# Turn 2 — the model "remembers" Japan
history.append(HumanMessage(content="What's the best time to go?"))
response = llm.invoke(history)
history.append(AIMessage(content=response.content))

# Turn 3 — still remembers the full context
history.append(HumanMessage(content="How much should I budget?"))
response = llm.invoke(history)
print(response.content)  # References Japan, timing, etc.
```

- **Why it matters:** This list *is* the model's memory. If you lose it, the conversation resets. If it grows too large, you hit token limits.

## 3. LangChain ChatMessageHistory (The Built-In Store)
**What it is:** LangChain provides dedicated classes to manage conversation history without manually appending messages.
- **How it works:** The history object automatically handles adding user messages and AI responses in the correct order.
- **Example:**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

# Create a history store
store = InMemoryChatMessageHistory()

# Add messages
store.add_message(HumanMessage(content="What is LangChain?"))
store.add_message(AIMessage(content="It's a framework for LLM apps."))
store.add_message(HumanMessage(content="Is it open source?"))
store.add_message(AIMessage(content="Yes, it's fully open source."))

# Retrieve all messages
print(store.messages)
# [HumanMessage(content='What is LangChain?'),
#  AIMessage(content="It's a framework for LLM apps."),
#  HumanMessage(content='Is it open source?'),
#  AIMessage(content="Yes, it's fully open source.")]

# Clear the history
store.clear()
print(store.messages)  # []
```

- **Why it matters:** It abstracts away manual list management and provides a clean interface for adding, retrieving, and clearing messages. It's the building block for more advanced memory systems.

## 4. RunnableWithMessageHistory (The Modern Pattern)
**What it is:** LangChain's recommended way to automatically inject conversation history into any chain or model call. It handles loading, saving, and scoping history per session.
- **How it works:** You wrap your LLM or chain with `RunnableWithMessageHistory`. It automatically:
    1. Loads the history for a given `session_id`
    2. Prepends it to the current prompt
    3. Saves the new messages after the response
- **Example:**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Dictionary to store histories per session
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the LLM with history management
with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Turn 1
response1 = with_history.invoke(
    {"input": "Remember: my favorite color is teal."},
    config={"configurable": {"session_id": "user_123"}}
)
print(response1.content)

# Turn 2 — different session, no memory
response2 = with_history.invoke(
    {"input": "What is my favorite color?"},
    config={"configurable": {"session_id": "user_999"}}
)
print(response2.content)  # "I don't know your favorite color."

# Turn 3 — same session, remembers!
response3 = with_history.invoke(
    {"input": "What is my favorite color?"},
    config={"configurable": {"session_id": "user_123"}}
)
print(response3.content)  # "Your favorite color is teal."
```

- **Why it matters:** This is production-ready memory. Each user gets their own isolated conversation history via `session_id`, and you never manually manage the message list.

## 5. Context Window Overflow (The Token Problem)
**What it is:** As conversations grow longer, the message history eventually exceeds the model's context window (e.g., 128k tokens). At that point, the API call fails or silently drops information.
- **How it works:** Every message in the history consumes tokens. A 50-turn conversation can easily consume 20,000+ tokens. If your model has an 8k context window, you'll hit the limit fast.
- **The Math:**

```
Average message ≈ 50-100 tokens
20 turns (40 messages) ≈ 2,000-4,000 tokens
100 turns (200 messages) ≈ 10,000-20,000 tokens
- System prompt + Current question
= Potential overflow
```

- **Why it matters:** Without a strategy to manage growing history, your chatbot will either crash, lose old context, or become extremely expensive as token counts balloon.

## 6. Trimming Strategies (Keeping History Manageable)
**What it is:** Techniques to automatically reduce the size of the conversation history so it stays within token limits while preserving the most important context.
- **Strategy A: Window Trimming (Keep Last N Turns)**

```python
from langchain_core.messages import trim_messages

# Keep only the last 10 messages
trimmed = trim_messages(
    store.messages,
    max_tokens=2000,
    strategy="last",       # Keep the most recent messages
    token_counter=len,     # Simple length-based count
    include_system=True,   # Always keep the system message
    allow_partial=False    # Don't cut messages in half
)
```

- **Strategy B: Token-Based Trimming**

```python
from langchain_openai import ChatOpenAI

# Use the actual model's tokenizer for accurate counting
trimmed = trim_messages(
    store.messages,
    max_tokens=4000,
    strategy="last",
    token_counter=ChatOpenAI(model="gpt-4o"),  # Accurate token count
    include_system=True
)
```

- **Strategy C: Summarization (Compress Old History)**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def summarize_history(history: list) -> str:
    summary_prompt = f"""
    Summarize the following conversation concisely.
    Preserve key facts, decisions, and user preferences.

    Conversation:
    {history}

    Summary:
    """
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content

# Replace old messages with a summary
old_messages = store.messages[:-10]  # Everything except last 10
summary = summarize_history(old_messages)

# Rebuild history: System + Summary + Recent messages
new_history = [
    SystemMessage(content=f"Previous conversation summary: {summary}")
] + store.messages[-10:]
```

- **Comparison:**

| Strategy | Pros | Cons |
|---|---|---|
| Window (Last N) | Simple, fast, cheap | Loses early context permanently |
| Token-based | Precise limit control | Still loses old messages |
| Summarization | Preserves key info | Costs extra LLM call, lossy |

- **Why it matters:** Choosing the right strategy depends on your use case. Customer support might need full history (summarization). A quick Q&A bot only needs the last few turns (window).

## 7. Persistence (Beyond In-Memory)
**What it is:** Storing conversation history in a database so it survives server restarts, scales across multiple users, and enables long-term memory.
- **How it works:** Replace `InMemoryChatMessageHistory` with a database-backed implementation.
- **Common Backends:**

```python
# Redis (fast, session-based)
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_session_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")

# PostgreSQL (durable, production)
from langchain_community.chat_message_histories import PostgresChatMessageHistory

def get_session_history(session_id: str):
    return PostgresChatMessageHistory(
        session_id,
        connection_string="postgresql://user:pass@localhost/dbname"
    )

# DynamoDB, MongoDB, SQLite, etc. also supported
```

- **Why it matters:** `InMemoryChatMessageHistory` is great for development but loses everything when the process restarts. Production apps need durable storage so users can resume conversations days later.

## 8. Best Practices (Production Patterns)
**What it is:** Guidelines for managing conversation history reliably in real applications.
- **Key Principles:**

| Practice | Reason |
|---|---|
| Always set `include_system=True` when trimming | Prevents losing core instructions |
| Use `session_id` per user/conversation | Isolates histories, prevents cross-contamination |
| Set a `max_tokens` budget for history | Leaves room for the current question + response |
| Log history to a database, not just memory | Enables debugging, analytics, and continuity |
| Summarize after N turns (e.g., every 20) | Prevents unbounded growth while preserving context |
| Clear history on topic change | Avoids confusing the model with irrelevant past context |
| Monitor token usage per request | Catch runaway costs before they hit your bill |

- **Token Budget Formula:**

```
Available Context Window
- System Prompt tokens
- History tokens (managed)
- Current User Input tokens
- Reserved Output tokens (max_tokens)
= Must be ≥ 0
```

- **Example Budget:**

```
Model context:        128,000 tokens
System prompt:          500 tokens
Reserved for output:  1,000 tokens
Current input:          200 tokens
─────────────────────────────────
Available for history: 126,300 tokens
```

- **Why it matters:** Without a token budget, a long conversation will silently exceed limits, causing API errors or truncated responses. Planning your token allocation upfront prevents production surprises.

## Quick Reference Cheat Sheet

| Concept | Tool / Class | Key Method |
|---|---|---|
| Store messages manually | Python `list` | `.append()` |
| Built-in history store | `InMemoryChatMessageHistory` | `.add_message()`, `.messages`, `.clear()` |
| Auto-inject history | `RunnableWithMessageHistory` | Wrap any runnable |
| Scope per user | `session_id` in config | `{"configurable": {"session_id": "..."}}` |
| Trim by count/tokens | `trim_messages()` | `strategy="last"`, `max_tokens=N` |
| Compress old history | LLM summarization | Custom function |
| Persist to database | `RedisChatMessageHistory`, `PostgresChatMessageHistory` | Connection string |
| Count tokens accurately | Pass model to `token_counter` | `token_counter=ChatOpenAI(...)` |

---

← [7. Structured Outputs](07-structured-outputs.md) · [Contents](README.md) · [9. Streaming and Async](09-streaming-and-async.md) →
