# 3. LLM APIs and SDKs

[Home](README.md) · Part 3 of 14

> The plumbing between your code and a provider: requests, messages, responses, keys, and rate limits.

**In this part:** LLM APIs · SDKs · Requests · Messages · Responses · API keys · .env files · Rate limits & errors

---

## 1. LLM APIs (The Service Counter)
**What it is:** An API, or Application Programming Interface, is the official doorway through which your code talks to an LLM provider, like OpenAI, Anthropic, Google, or Mistral.
- **How it works:** Instead of typing into a chat interface, your code sends a structured digital request to the provider's server. The server processes the request, runs the LLM, and sends back a structured response.
- **Analogy:** If the chat interface is ordering food by talking to a waiter, the API is placing the order through a standardized digital kiosk.
- **Examples:**
    - Sending a request to `https://api.openai.com/v1/chat/completions`
    - Sending a request to `https://api.anthropic.com/v1/messages`
- **Why it matters:** APIs allow developers to build custom applications, automate tasks, integrate AI into websites, and process data at scale.

## 2. SDKs (The Developer Toolkit)
**What it is:** An SDK, or Software Development Kit, is a pre-built library that makes using the API much easier. Instead of writing raw network code, you use simple functions provided by the SDK.
- **How it works:** The SDK handles the messy details for you: formatting the JSON, making the HTTP request, handling errors, and parsing the response.
- **Analogy:** If the API is a complex engine, the SDK is the steering wheel, pedals, and dashboard that make it easy to drive.
- **Examples:**
    - Python: `import openai` or `import anthropic`
    - JavaScript: `import OpenAI from 'openai'`
- **Why it matters:** SDKs reduce boilerplate code, prevent common mistakes, and make your code cleaner and easier to maintain.

## 3. Requests (The Order Form)
**What it is:** The structured message your code sends to the LLM provider. It contains all the instructions, settings, and context the model needs.
- **How it works:** Most LLM APIs expect a JSON object. This object includes the model name, the conversation history, and generation settings.
- **Common Fields:**
    - `model`: Which AI to use (e.g., `"gpt-4o"`, `"claude-sonnet-4-5"`).
    - `messages`: The conversation history.
    - `temperature`: Creativity level.
    - `max_tokens`: Maximum length of the response.
    - `stream`: Whether to return the full answer at once, or stream it word-by-word.
- **Example:**

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "Explain APIs in one sentence."}
  ],
  "temperature": 0.7,
  "max_tokens": 150
}
```

- **Why it matters:** A malformed request will result in an error. A well-structured request ensures you get exactly the behavior you want.

## 4. Messages (The Conversation History)
**What it is:** The core of the request. It is an array of objects representing the conversation so far. LLMs are stateless; they don't remember previous conversations unless you send the history back to them.
- **How it works:** Each message has a `role` and `content`.
    - `system`: Sets the behavior/persona.
    - `user`: What the human asked.
    - `assistant`: What the AI previously answered.
- **Example:**

```json
[
  {"role": "system", "content": "You are a helpful pirate."},
  {"role": "user", "content": "Where is the treasure?"},
  {"role": "assistant", "content": "X marks the spot, matey!"},
  {"role": "user", "content": "Which map should I use?"}
]
```

- **Why it matters:** The model uses the entire `messages` array to understand context. If you don't include the previous messages, the model will forget what you were just talking about.

## 5. Responses (The Delivery Package)
**What it is:** The structured data sent back by the LLM provider after processing your request.
- **How it works:** It isn't just plain text. It is usually a JSON object containing metadata, usage statistics, and the actual generated content.
- **Common Fields:**
    - `id`: A unique identifier for the request.
    - `choices`: The actual generated text.
    - `usage`: How many tokens were used for the prompt and completion.
    - `finish_reason`: Why the model stopped (e.g., `"stop"` for natural end, `"length"` because it hit the token limit).
- **Example:**

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Use the old parchment map with the red compass."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 12,
    "total_tokens": 54
  }
}
```

- **Why it matters:** You need to parse the response to extract the text. You also need to monitor `usage` to track costs and `finish_reason` to know if the response was cut off.

## 6. API Keys (The VIP Pass)
**What it is:** A secret string of characters that proves who you are and authorizes your application to use the provider's services.
- **How it works:** You include the API key in the header of every HTTP request, usually as `Authorization: Bearer sk-...`.
- **Analogy:** It is like a password, a credit card, and a security badge all rolled into one.
- **Examples:**
    - `sk-proj-abc123...`
    - `ANTHROPIC_API_KEY=sk-ant-api03...`
- **Why it matters:** It tracks your usage, bills your account, and enforces rate limits. If someone steals your API key, they can run up charges on your account or access your data.

## 7. `.env` Files (The Secret Vault)
**What it is:** A special file used to store environment variables, most commonly your API keys and configuration settings.
- **How it works:** Instead of hardcoding your API key directly into your Python or JavaScript file, you store it in a `.env` file. Your code reads the key from the environment when it runs.
- **Example `.env` file:**

```env
OPENAI_API_KEY=sk-proj-abc123...
DATABASE_URL=postgres://localhost:5432/mydb
```

- **Example Python code reading it:**

```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

- **Why it matters:**
    - **Security:** You never want to upload your API keys to GitHub.
    - **Best Practice:** Always add `.env` to your `.gitignore` file so it is never accidentally committed to version control.

## 8. Rate Limits & Errors (The Bouncer)
**What it is:** Constraints placed by the API provider to prevent abuse and manage server load.
- **How it works:** Providers limit how many requests you can make per minute or how many tokens you can process per day. If you exceed them, the API returns an error.
- **Common Errors:**
    - `401 Unauthorized`: Invalid or missing API key.
    - `429 Too Many Requests`: You hit a rate limit. You need to slow down or upgrade your plan.
    - `500 Internal Server Error`: The provider's servers are having issues.
    - `400 Bad Request`: Your JSON is malformed or you requested an invalid model.
- **Why it matters:** Production applications must handle these errors gracefully, usually by implementing "retry logic with exponential backoff" (waiting a little bit and trying again).

## Summary: The Flow of an LLM API Call
1.  **Load** your API key from the `.env` file.
2.  **Initialize** the SDK client.
3.  **Build** a request containing the `model`, `messages`, and settings.
4.  **Send** the request to the API endpoint.
5.  **Receive** the JSON response.
6.  **Extract** the generated text and display it to the user.

---

← [2. Prompt Engineering](02-prompt-engineering.md) · [Contents](README.md) · [4. LangChain Basics](04-langchain-basics.md) →
