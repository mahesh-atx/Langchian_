# 6. Output Handling

[Home](README.md) · Part 6 of 14

> What actually comes back from `.invoke()` and how to read it safely.

**In this part:** Raw responses · Extracting content · Metadata, usage & errors · Invalid or unexpected output

---

## 1. Understanding Raw LLM Responses (The Full Payload)
**What it is:** When an LLM API returns a result, it doesn't just send back plain text. It returns a structured object containing the generated text *plus* metadata about how the generation happened.
- **How it works:** Whether you're using raw HTTP, an SDK, or LangChain, the response object typically contains:
    - The actual generated text (the "content")
    - Token usage statistics (how much you'll be billed)
    - A finish reason (why the model stopped generating)
    - Model metadata (which model actually ran)
    - An ID for tracking the request
- **Raw API Response Example (JSON):**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "gpt-4o-2024-08-06",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

- **Why it matters:** If you only grab the text, you miss critical information for billing, debugging, rate-limiting, and detecting when the model cut off early.

## 2. Extracting Response Content (Getting the Text)
**What it is:** Pulling the actual generated text out of the response object so you can display it, store it, or pass it to the next step in your pipeline.
- **How it works:** The method depends on whether you're using raw APIs, SDKs, or LangChain.
- **LangChain Example:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("What is the capital of France?")

# The response is an AIMessage object
print(type(response))        # <class 'langchain_core.messages.AIMessage'>
print(response.content)      # "The capital of France is Paris."
print(response.content[:10]) # "The capita" (string slicing works)
```

- **Raw OpenAI SDK Example:**

```python
import openai
client = openai.OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)

# Navigate the nested structure
text = response.choices[0].message.content
print(text)  # "The capital of France is Paris."
```

- **Why it matters:** In LangChain, `response.content` is always a string. In raw APIs, you must navigate nested dictionaries/objects. Knowing the exact path prevents `NoneType` errors and broken pipelines.

## 3. Metadata, Usage, and Errors (The Hidden Data)
**What it is:** The non-text information attached to every response that tells you *how* the model generated the answer, *how much* it cost, and *why* it stopped.
- **Key Fields:**

| Field | Meaning |
|---|---|
| `response.content` | The generated text |
| `response.response_metadata` | Provider-specific metadata (model name, finish reason, etc.) |
| `response.usage_metadata` | Token counts (input, output, total) |
| `response.id` | Unique request ID for debugging |
| `finish_reason` | Why generation stopped (`"stop"`, `"length"`, `"tool_calls"`) |

- **LangChain Example:**

```python
response = llm.invoke("Explain gravity briefly.")

# Token usage (critical for billing)
print(response.usage_metadata)
# {'input_tokens': 11, 'output_tokens': 24, 'total_tokens': 35}

# Finish reason (did it complete or get cut off?)
print(response.response_metadata)
# {'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', ...}

# Check if it was cut off due to length
if response.response_metadata.get('finish_reason') == 'length':
    print("WARNING: Response was truncated!")
```

- **Error Handling:**

```python
from langchain_core.exceptions import OutputParserException
from openai import RateLimitError, APIConnectionError, BadRequestError

try:
    response = llm.invoke("Do something")
except RateLimitError:
    print("Too many requests. Backing off...")
except APIConnectionError:
    print("Network error. Check your connection.")
except BadRequestError as e:
    print(f"Invalid request: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

- **Why it matters:** Without monitoring `finish_reason`, you might silently accept truncated answers. Without tracking `usage_metadata`, you can't control costs. Without error handling, one API timeout crashes your entire application.

## 4. Handling Invalid or Unexpected Output (Guardrails)
**What it is:** Strategies for detecting and recovering when the LLM produces garbage, refuses to answer, returns the wrong format, or hallucinates.
- **Common Failure Modes:**
    - **Empty/Null Content:** The model returns nothing.
    - **Wrong Format:** You asked for JSON, got a paragraph.
    - **Refusal:** The model says "I can't help with that."
    - **Truncation:** The response cuts off mid-sentence.
    - **Hallucination:** Confident but factually wrong output.

- **Strategy 1: Validate Content Exists**

```python
response = llm.invoke("Generate a product name.")

if not response.content or response.content.strip() == "":
    print("ERROR: Empty response received. Retrying...")
    # Retry logic here
```

- **Strategy 2: Enforce Structured Output (JSON Mode)**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

# Define expected schema
class Product(BaseModel):
    name: str
    price: float
    category: str

# Bind the schema to the model
llm = ChatOpenAI(model="gpt-4o")
structured_llm = llm.with_structured_output(Product)

try:
    result = structured_llm.invoke("Create a product for a smart water bottle.")
    print(result.name)      # "HydroTrack Pro"
    print(result.price)     # 49.99
    print(result.category)  # "Fitness"
except Exception as e:
    print(f"Failed to parse structured output: {e}")
```

- **Strategy 3: Retry with Exponential Backoff**

```python
import time

def safe_invoke(llm, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = llm.invoke(prompt)
            if response.content and len(response.content) > 10:
                return response.content
            else:
                print(f"Attempt {attempt+1}: Empty/short response. Retrying...")
        except Exception as e:
            print(f"Attempt {attempt+1}: Error - {e}")
            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
    return None  # All retries failed

result = safe_invoke(llm, "Summarize this article...")
```

- **Strategy 4: Detect Refusals and Hallucinations**

```python
refusal_phrases = ["I cannot", "I'm sorry, but", "I don't have information"]

response = llm.invoke("What is the secret formula for Coca-Cola?")

# Check for refusal
if any(phrase in response.content.lower() for phrase in refusal_phrases):
    print("Model refused to answer. Falling back to default.")

# Check for truncation
if response.response_metadata.get('finish_reason') == 'length':
    print("Response was cut off. Consider increasing max_tokens.")
```

- **Why it matters:** In production, LLMs fail unpredictably. Without guardrails, one malformed response can crash your app, corrupt your database, or show garbage to your users. Defensive output handling is the difference between a demo and a reliable product.

## Quick Reference Cheat Sheet

| Task | LangChain Approach | Key Attribute |
|---|---|---|
| Get text | `response.content` | String |
| Get token usage | `response.usage_metadata` | Dict with `input_tokens`, `output_tokens` |
| Get finish reason | `response.response_metadata['finish_reason']` | `"stop"`, `"length"`, `"tool_calls"` |
| Force JSON output | `llm.with_structured_output(Schema)` | Pydantic model |
| Handle errors | `try/except` around `.invoke()` | `RateLimitError`, `BadRequestError` |
| Retry failures | Loop with `time.sleep(2 ** attempt)` | Exponential backoff |
| Detect empty output | `if not response.content` | Guard clause |

---

← [5. Messages and Prompt Templates](05-messages-and-prompt-templates.md) · [Contents](README.md) · [7. Structured Outputs](07-structured-outputs.md) →
