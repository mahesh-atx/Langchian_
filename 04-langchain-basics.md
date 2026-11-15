# 4. LangChain Basics

[Home](README.md) · Part 4 of 14

> One interface for every provider: install, import, and swap models with `init_chat_model`.

**In this part:** Overview · Installation · langchain vs langchain-core · init_chat_model · OpenAI · Gemini · Anthropic · Groq

---

## 1. LangChain Overview (The Framework)
**What it is:** LangChain is an open-source Python/JavaScript framework that simplifies building applications powered by LLMs. It provides standardized tools to connect to dozens of AI providers, manage prompts, chain steps together, and handle memory.
- **How it works:** Instead of writing custom API code for every AI provider, LangChain gives you a unified interface. You swap models, add tools, or build complex pipelines using consistent patterns.
- **Why it matters:** It turns "calling an API" into "building an AI application" with routing, memory, agents, and multi-step reasoning.

## 2. Installation (Getting Started)
**What it is:** Installing the necessary Python packages via pip.
- **How it works:** LangChain is modular. You install the core framework, plus individual "connector" packages for each AI provider you want to use.
- **The Commands:**

```bash
# Core framework
pip install langchain

# Provider-specific connectors
pip install langchain-openai       # OpenAI / Azure
pip install langchain-anthropic    # Claude
pip install langchain-google-genai # Gemini
pip install langchain-groq         # Groq (fast inference)

# Environment variable management
pip install python-dotenv
```

- **Why it matters:** You only install what you need. If you only use Anthropic, you don't need the OpenAI or Gemini packages.

## 3. langchain vs langchain-core (The Architecture)
**What it is:** LangChain is split into layered packages to keep things lightweight and modular.
- **`langchain-core`:** The absolute foundation. Contains base classes, interfaces, and abstractions (like what a "ChatModel" or "Prompt" *is*). It has zero provider dependencies.
- **`langchain`:** The main orchestration layer. Contains chains, agents, and retrieval strategies. It depends on `langchain-core`.
- **`langchain-[provider]`:** The specific connectors (e.g., `langchain-openai`). These depend on `langchain-core` and the provider's official SDK.
- **The Dependency Tree:**

```
langchain-core  (abstract interfaces)
     ↑
langchain-openai, langchain-anthropic, etc.  (provider logic)
     ↑
langchain  (orchestration, chains, agents)
```

- **Why it matters:** Understanding this prevents version conflicts. If you get import errors, it's usually because a provider package is out of sync with `langchain-core`.

## 4. init_chat_model (The Unified Interface)
**What it is:** A modern LangChain utility that lets you instantiate *any* supported chat model using a single function, without importing provider-specific classes.
- **How it works:** You pass the model name as a string, and LangChain automatically figures out which provider package to use behind the scenes.
- **Example:**

```python
from langchain.chat_models import init_chat_model

# Just change the string to swap providers entirely
llm = init_chat_model("gpt-4o", model_provider="openai", temperature=0)
llm = init_chat_model("claude-3-5-sonnet-20241022", model_provider="anthropic", temperature=0)
llm = init_chat_model("gemini-1.5-pro", model_provider="google-genai", temperature=0)
llm = init_chat_model("llama-3.1-70b-versatile", model_provider="groq", temperature=0)
```

- **Why it matters:** It makes switching between providers trivial. You can A/B test GPT-4o vs Claude vs Gemini by changing one string, without rewriting imports or logic.

## 5. Model Invocation — OpenAI
**What it is:** Calling OpenAI models (GPT-4o, GPT-4o-mini, o1, etc.) through LangChain.
- **Setup:**

```python
# .env file
OPENAI_API_KEY=sk-proj-your-key-here
```

- **Invocation:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Simple string invocation
response = llm.invoke("Explain quantum computing in one sentence.")
print(response.content)

# Message-list invocation (for chat history)
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a sarcastic scientist."),
    HumanMessage(content="Why is the sky blue?")
]
response = llm.invoke(messages)
print(response.content)
```

- **Why it matters:** `ChatOpenAI` is the most battle-tested connector. It supports streaming, function calling, and JSON mode out of the box.

## 6. Model Invocation — Gemini
**What it is:** Calling Google's Gemini models through LangChain.
- **Setup:**

```python
# .env file
GOOGLE_API_KEY=your-google-api-key-here
```

- **Invocation:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

response = llm.invoke("Write a haiku about neural networks.")
print(response.content)
```

- **Why it matters:** Gemini models have massive context windows (up to 1M+ tokens) and native multimodal support (images, video, audio). The connector handles Google's specific authentication and safety settings.

## 7. Model Invocation — Anthropic
**What it is:** Calling Anthropic's Claude models through LangChain.
- **Setup:**

```python
# .env file
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

- **Invocation:**

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.7)

response = llm.invoke("Summarize the plot of Hamlet in 3 bullet points.")
print(response.content)
```

- **Why it matters:** Claude models excel at long-form reasoning, coding, and following complex system prompts. The connector supports Anthropic's unique features like extended thinking and tool use.

## 8. Model Invocation — Groq
**What it is:** Calling models hosted on Groq's ultra-fast LPU (Language Processing Unit) hardware. Groq doesn't make its own models; it hosts open-source models (Llama, Mixtral, Gemma) at blazing speed.
- **Setup:**

```python
# .env file
GROQ_API_KEY=gsk_your-key-here
```

- **Invocation:**

```python
from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0.7)

response = llm.invoke("What are 3 benefits of open-source AI?")
print(response.content)
```

- **Why it matters:** Groq offers dramatically faster token generation (often 500+ tokens/second) compared to standard GPU inference. It's ideal for real-time applications, chatbots, and latency-sensitive pipelines where speed matters more than proprietary model features.

## 9. Loading Keys with .env (Tying It Together)
**What it is:** The standard pattern for securely loading all your API keys before invoking any model.
- **The Pattern:**

```python
import os
from dotenv import load_dotenv

# Load all keys from .env into environment variables
load_dotenv()

# Now LangChain connectors auto-detect them
# No need to pass api_key= manually
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")  # Reads OPENAI_API_KEY automatically
```

- **Why it matters:** LangChain connectors are designed to look for standard environment variable names (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) automatically. Using `.env` + `load_dotenv()` keeps your code clean and your secrets safe.

## Quick Reference Cheat Sheet

| Provider | Package | Class | Env Variable | Example Model |
|---|---|---|---|---|
| OpenAI | `langchain-openai` | `ChatOpenAI` | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `langchain-anthropic` | `ChatAnthropic` | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20241022` |
| Gemini | `langchain-google-genai` | `ChatGoogleGenerativeAI` | `GOOGLE_API_KEY` | `gemini-1.5-pro` |
| Groq | `langchain-groq` | `ChatGroq` | `GROQ_API_KEY` | `llama-3.1-70b-versatile` |

---

← [3. LLM APIs and SDKs](03-llm-apis-and-sdks.md) · [Contents](README.md) · [5. Messages and Prompt Templates](05-messages-and-prompt-templates.md) →
