# 9. Streaming and Async

[Home](README.md) · Part 9 of 14

> Showing tokens as they are produced, and running calls concurrently with async and batching.

**In this part:** Why streaming matters · Token-by-token streaming · .stream() · .astream() · FastAPI endpoints · Async & batching · Error handling · Best practices

---

## 1. The UX Problem (Why Streaming Matters)
**What it is:** By default, an LLM generates its entire response on the server before sending anything back to the client. For long responses, this means the user stares at a blank screen for 5–30 seconds before seeing any text.
- **Without Streaming:**

```
User sends prompt...
[Waiting 12 seconds...]
[Waiting 12 seconds...]
[Full response appears all at once]
```

- **With Streaming:**

```
User sends prompt...
"The" (50ms)
"The capital" (100ms)
"The capital of France" (150ms)
"The capital of France is Paris." (200ms)
```

- **How it works:** LLMs generate text one token at a time. Streaming sends each token (or small chunk of tokens) to the client *as it's produced*, rather than waiting for the full response.
- **Why it matters:** Streaming dramatically improves perceived performance. Users feel the AI is "thinking" and "typing" in real time, reducing frustration and abandonment rates.

## 2. Streaming Basics (Token-by-Token)
**What it is:** The fundamental mechanism where the API returns a sequence of small chunks instead of one large response.
- **How it works:** Under the hood, the API uses Server-Sent Events (SSE) or chunked transfer encoding. Each chunk contains a small piece of the generated text (usually 1–5 tokens).
- **Raw OpenAI SDK Example:**

```python
import openai
client = openai.OpenAI()

# Set stream=True
stream = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a short poem about rain."}],
    stream=True  # Enable streaming
)

# Iterate over chunks as they arrive
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

- **Why it matters:** Understanding the raw mechanism helps you debug issues when LangChain's abstractions behave unexpectedly.

## 3. LangChain `.stream()` (Synchronous Streaming)
**What it is:** LangChain's built-in method to stream responses from any chat model. It returns a Python generator that yields chunks.
- **How it works:** You call `.stream()` instead of `.invoke()`. It yields `AIMessageChunk` objects that you can print, accumulate, or forward to a frontend.
- **Example:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Stream the response
print("AI: ", end="", flush=True)
for chunk in llm.stream("Explain how the internet works in 3 sentences."):
    print(chunk.content, end="", flush=True)
print()  # Newline at the end
```

- **Accumulating the Full Response:**

```python
full_response = ""

for chunk in llm.stream("Tell me a joke."):
    print(chunk.content, end="", flush=True)  # Display in real time
    full_response += chunk.content            # Build the full string

# After streaming is done, you have the complete text
print(f"\n\nFull response length: {len(full_response)} chars")
```

- **Why it matters:** `.stream()` is the simplest way to add real-time output to CLI tools, scripts, and backend services.

## 4. Async Streaming with `.astream()` (Non-Blocking)
**What it is:** The asynchronous version of streaming, using Python's `async/await` syntax. It allows your application to handle other tasks while waiting for tokens.
- **How it works:** Instead of blocking the thread while waiting for the next chunk, `async for` yields control back to the event loop. This is critical for web servers handling many concurrent users.
- **Example:**

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def stream_response(prompt: str):
    print("AI: ", end="", flush=True)
    async for chunk in llm.astream(prompt):
        print(chunk.content, end="", flush=True)
    print()

# Run the async function
asyncio.run(stream_response("What is machine learning?"))
```

- **Concurrent Streaming (Multiple Users):**

```python
async def handle_multiple_users():
    prompts = [
        "Explain Python.",
        "Explain JavaScript.",
        "Explain Rust."
    ]

    # Stream all three simultaneously
    tasks = [stream_response(p) for p in prompts]
    await asyncio.gather(*tasks)

asyncio.run(handle_multiple_users())
```

- **Why it matters:** In a web server with 1,000 concurrent users, synchronous streaming would block the server. Async streaming lets one thread serve all users simultaneously.

## 5. Streaming in Web Frameworks (FastAPI)
**What it is:** Exposing a streaming LLM response as an HTTP endpoint so a frontend (React, Vue, etc.) can display tokens in real time.
- **How it works:** FastAPI's `StreamingResponse` sends data to the browser as a stream of chunks using Server-Sent Events (SSE).
- **Complete FastAPI Example:**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI
import asyncio

app = FastAPI()
llm = ChatOpenAI(model="gpt-4o")

@app.get("/stream")
async def stream_endpoint(prompt: str):
    async def generate():
        async for chunk in llm.astream(prompt):
            if chunk.content:
                # SSE format: "data: {text}\n\n"
                yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

- **Frontend JavaScript (Browser):**

```javascript
const eventSource = new EventSource("/stream?prompt=Tell me a story");

eventSource.onmessage = (event) => {
    if (event.data === "[DONE]") {
        eventSource.close();
        return;
    }
    // Append each chunk to the page in real time
    document.getElementById("output").textContent += event.data;
};
```

- **Why it matters:** This is the exact pattern used by ChatGPT, Claude, and every major AI chat interface. The browser receives tokens as they're generated, creating the "typing" effect.

## 6. Streaming with LangChain Chains (Complex Pipelines)
**What it is:** Streaming doesn't just work on raw models. You can stream the output of entire chains, including prompt templates, retrievers, and output parsers.
- **How it works:** LangChain's `.stream()` and `.astream()` methods work on any `Runnable`, including `ChatPromptTemplate | ChatModel` pipelines.
- **Example:**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# Build a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}")
])

chain = prompt | llm

# Stream the entire chain
for chunk in chain.stream({"role": "pirate", "question": "What is treasure?"}):
    print(chunk.content, end="", flush=True)
```

- **Streaming Modes (for complex chains):**

```python
# Stream only the final output
for chunk in chain.stream(input, stream_mode="values"):
    print(chunk)

# Stream intermediate steps (useful for debugging)
for event in chain.stream(input, stream_mode="updates"):
    print(event)  # Shows each step's output
```

- **Why it matters:** In production, you rarely call a raw model. You call a chain. Streaming must work through the entire pipeline, not just the final LLM call.

## 7. Async Invocation and Batching (Beyond Streaming)
**What it is:** Async isn't just for streaming. LangChain provides async versions of all core methods for non-blocking execution.
- **Key Methods:**

| Sync Method | Async Method | Use Case |
|---|---|---|
| `.invoke()` | `.ainvoke()` | Single non-blocking call |
| `.stream()` | `.astream()` | Non-blocking streaming |
| `.batch()` | `.abatch()` | Multiple concurrent calls |

- **Async Batch Example:**

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

async def batch_process():
    prompts = [
        "What is Python?",
        "What is Java?",
        "What is Go?"
    ]

    # Process all 3 concurrently (much faster than sequential)
    results = await llm.abatch(prompts)

    for prompt, result in zip(prompts, results):
        print(f"Q: {prompt}")
        print(f"A: {result.content[:100]}...\n")

asyncio.run(batch_process())
```

- **Why it matters:** If you need to process 100 prompts, sequential `.invoke()` takes 100x the time of one call. Async `.abatch()` processes them concurrently, reducing total time dramatically.

## 8. Handling Streaming Errors and Edge Cases
**What it is:** Streaming introduces unique failure modes that don't exist with synchronous responses.
- **Common Issues:**
    - **Connection drops mid-stream:** The API connection fails after 50% of tokens.
    - **Empty chunks:** Some chunks have `content=None` (metadata only).
    - **Rate limits during stream:** The provider throttles you mid-generation.
    - **Client disconnects:** The user closes the browser before the stream finishes.
- **Defensive Pattern:**

```python
async def safe_stream(llm, prompt, max_retries=2):
    for attempt in range(max_retries):
        try:
            full_response = ""
            async for chunk in llm.astream(prompt):
                if chunk.content:  # Skip empty/metadata chunks
                    yield chunk.content
                    full_response += chunk.content
            return  # Success
        except Exception as e:
            print(f"Stream error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                yield f"\n[Error: {str(e)}]"
            await asyncio.sleep(1)
```

- **Why it matters:** In production, network hiccups, rate limits, and client disconnects are *expected*. Without error handling, a single dropped connection leaves your UI in a broken state.

## 9. Best Practices (Production Patterns)
**What it is:** Guidelines for implementing streaming reliably in real applications.
- **Key Principles:**

| Practice | Reason |
|---|---|
| Always use `flush=True` when printing | Ensures tokens appear immediately, not buffered |
| Use `async` for web servers | Prevents blocking other users' requests |
| Accumulate full response in background | You often need the complete text for logging/storage |
| Handle `[DONE]` signals | Know when the stream has finished cleanly |
| Set timeouts | Don't let a hung stream block forever |
| Use SSE for browser clients | Native browser support, no WebSocket complexity |
| Log token count from chunks | Track usage for billing even in streaming mode |

- **Streaming + Logging Pattern:**

```python
async def stream_and_log(llm, prompt, session_id):
    full_text = ""

    async for chunk in llm.astream(prompt):
        if chunk.content:
            yield chunk.content
            full_text += chunk.content

    # After stream completes, save to database
    await save_to_db(session_id, full_text)
```

- **Why it matters:** Streaming is a UX feature, but logging, billing, and error recovery are operational necessities. You need both happening simultaneously.

## Quick Reference Cheat Sheet

| Task | Method | Key Syntax |
|---|---|---|
| Sync streaming | `.stream()` | `for chunk in llm.stream(...)` |
| Async streaming | `.astream()` | `async for chunk in llm.astream(...)` |
| Async single call | `.ainvoke()` | `result = await llm.ainvoke(...)` |
| Async batch | `.abatch()` | `results = await llm.abatch([...])` |
| Stream a chain | `chain.stream()` | `(prompt \| llm).stream(input)` |
| Web endpoint | FastAPI `StreamingResponse` | `yield f"data: {chunk}\n\n"` |
| Browser client | `EventSource` (SSE) | `new EventSource("/stream")` |
| Accumulate text | String concatenation | `full += chunk.content` |
| Skip empty chunks | Guard clause | `if chunk.content:` |

---

← [8. Conversation History](08-conversation-history.md) · [Contents](README.md) · [10. Embeddings and Vector Stores](10-embeddings-and-vector-stores.md) →
