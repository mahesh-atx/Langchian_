# 5. Messages and Prompt Templates

[Home](README.md) · Part 5 of 14

> The three message roles, string templates, chat templates, variables, and reusable prompt objects.

**In this part:** Message types · PromptTemplate · ChatPromptTemplate · Variables in prompts · MessagesPlaceholder · Dynamic prompts · Reusing templates

---

## 1. Message Types (The Conversation Roles)
**What it is:** LangChain represents every piece of a conversation as a typed "Message" object. Each message has a specific `role` that tells the LLM how to interpret it.
- **The Three Core Types:**
    - `SystemMessage`: The hidden director's instructions. Sets persona, rules, and constraints.
    - `HumanMessage`: The user's input, question, or command.
    - `AIMessage`: The model's previous response (used to maintain conversation history).
- **How it works:** You build a list of these messages in chronological order and pass them to the model. The LLM reads them sequentially to understand the full context.
- **Example:**

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a helpful coding tutor."),
    HumanMessage(content="What is a for loop?"),
    AIMessage(content="A for loop repeats code over a sequence."),
    HumanMessage(content="Can you show me an example in Python?")
]

response = llm.invoke(messages)
print(response.content)
```

- **Why it matters:** Without typed messages, the LLM wouldn't know which text is an instruction, which is user input, and which is its own prior output. This structure is what enables multi-turn conversations.

## 2. PromptTemplate (String-Based Templates)
**What it is:** The simplest way to create reusable prompt strings with fill-in-the-blank variables. It produces a raw string, not a message list.
- **How it works:** You define a template string with `{variable}` placeholders. At runtime, you call `.format()` or `.invoke()` to inject the actual values.
- **Example:**

```python
from langchain_core.prompts import PromptTemplate

# Define the template
template = PromptTemplate(
    input_variables=["product", "audience"],
    template="Write a catchy slogan for {product} targeted at {audience}."
)

# Fill in the variables
formatted = template.format(product="energy drinks", audience="college students")
print(formatted)
# Output: "Write a catchy slogan for energy drinks targeted at college students."

# Or use .invoke() to get a PromptValue object
prompt_value = template.invoke({"product": "energy drinks", "audience": "college students"})
```

- **Why it matters:** It separates the *structure* of your prompt from the *data*. You can reuse the same template thousands of times with different inputs without rewriting the prompt logic.

## 3. ChatPromptTemplate (Message-Based Templates)
**What it is:** The modern, standard way to build prompts for chat models. Instead of a single string, it constructs a full list of typed messages (System, Human, AI).
- **How it works:** You define a list of message tuples: `(role, template_string)`. LangChain converts these into proper Message objects when invoked.
- **Example:**

```python
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert {specialty} chef."),
    ("human", "Suggest a recipe using {ingredient}.")
])

# Invoke with variables
messages = chat_template.invoke({"specialty": "Italian", "ingredient": "eggplant"})

# This produces:
# [SystemMessage(content="You are an expert Italian chef."),
#  HumanMessage(content="Suggest a recipe using eggplant.")]

response = llm.invoke(messages)
```

- **Why it matters:** Chat models expect a list of messages, not a flat string. `ChatPromptTemplate` ensures your system instructions, user inputs, and conversation history are all properly structured and typed for the API.

## 4. Using Variables in Prompts (Dynamic Injection)
**What it is:** The `{variable}` syntax that allows you to inject dynamic data into your prompts at runtime.
- **How it works:** Any word wrapped in curly braces `{}` inside a template string becomes a required input variable. When you call `.invoke()` or `.format()`, you must provide a dictionary with matching keys.
- **Advanced Example:**

```python
from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a {tone} assistant specializing in {domain}."),
    ("human", "Explain {concept} to a {level} student in under {word_limit} words.")
])

# Inject all variables at once
messages = template.invoke({
    "tone": "friendly",
    "domain": "physics",
    "concept": "black holes",
    "level": "high school",
    "word_limit": "100"
})
```

- **Why it matters:** Variables make prompts scalable. One template can serve millions of unique requests. It also prevents "prompt injection" risks by keeping user data separate from instruction structure.

## 5. MessagesPlaceholder (Inserting Dynamic Message Lists)
**What it is:** A special placeholder inside a `ChatPromptTemplate` that allows you to inject an entire *list* of messages dynamically at a specific position.
- **How it works:** Instead of a single string, you reserve a "slot" in your template. At runtime, you pass a list of Message objects into that slot.
- **Example:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # Dynamic slot
    ("human", "{current_question}")
])

# Simulate a conversation history
history = [
    HumanMessage(content="What is LangChain?"),
    AIMessage(content="It's a framework for building LLM apps."),
    HumanMessage(content="Is it free?"),
    AIMessage(content="Yes, it's open-source.")
]

messages = template.invoke({
    "chat_history": history,
    "current_question": "How do I install it?"
})
```

- **Why it matters:** This is the backbone of **memory** in chatbots. It lets you append the entire conversation history into the prompt dynamically, so the AI remembers context without you hardcoding every message.

## 6. Creating Dynamic Prompts (Conditional Logic)
**What it is:** Building prompts that change their structure, instructions, or content based on runtime conditions, user input, or external data.
- **How it works:** You use Python logic to select different templates, inject different variables, or conditionally add messages before invoking the model.
- **Example:**

```python
from langchain_core.prompts import ChatPromptTemplate

def build_prompt(user_type, query):
    if user_type == "beginner":
        system_msg = "Explain everything in simple terms. Avoid jargon."
    elif user_type == "expert":
        system_msg = "Provide technical depth. Include code examples."
    else:
        system_msg = "Be concise and helpful."

    template = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", "{query}")
    ])
    return template.invoke({"query": query})

# Different users get different prompts
beginner_prompt = build_prompt("beginner", "What is recursion?")
expert_prompt = build_prompt("expert", "What is recursion?")
```

- **Why it matters:** Real applications aren't one-size-fits-all. Dynamic prompts let you personalize AI behavior per user, per task, or per context without maintaining hundreds of separate templates.

## 7. Reusing Prompt Templates (Composition & Partials)
**What it is:** Techniques to avoid repetition by building modular, composable prompt templates that can be shared, inherited, or partially pre-filled.
- **How it works:**
    - **Partial Variables:** Pre-fill some variables in advance, leaving others for runtime.
    - **Composition:** Combine smaller templates into larger ones.
    - **Pipeline Chaining:** Pass the output of one template as input to another.
- **Example — Partial Variables:**

```python
from langchain_core.prompts import ChatPromptTemplate

# Create a template with 3 variables
full_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} at {company}."),
    ("human", "{question}")
])

# Pre-fill the company name (partial)
company_template = full_template.partial(company="Acme Corp")

# Now you only need to provide 'role' and 'question' at runtime
messages = company_template.invoke({"role": "engineer", "question": "What's our stack?"})
```

- **Example — Composition:**

```python
from langchain_core.prompts import ChatPromptTemplate, PipelinePromptTemplate

# Define reusable sub-prompts
identity_prompt = ChatPromptTemplate.from_template("You are {name}, a {role}.")
task_prompt = ChatPromptTemplate.from_template("Your task: {task}")

# Combine them into a pipeline
full_prompt = ChatPromptTemplate.from_messages([
    ("system", "{identity}\n\n{task}"),
    ("human", "{user_input}")
])

messages = full_prompt.invoke({
    "identity": identity_prompt.format(name="Ada", role="mathematician"),
    "task": task_prompt.format(task="Solve the equation."),
    "user_input": "What is 2 + 2?"
})
```

- **Why it matters:** In production systems, you might have dozens of prompts sharing the same system instructions, safety disclaimers, or formatting rules. Reusability ensures consistency, reduces errors, and makes updates trivial (change one place, update everywhere).

## Quick Reference Cheat Sheet

| Component | Purpose | Key Method |
|---|---|---|
| `SystemMessage` | Set rules/persona | Pass in message list |
| `HumanMessage` | User input | Pass in message list |
| `AIMessage` | Model's prior output | Pass in message list |
| `PromptTemplate` | Simple string template | `.format()` / `.invoke()` |
| `ChatPromptTemplate` | Multi-message template | `.from_messages()` / `.invoke()` |
| `MessagesPlaceholder` | Inject dynamic history | `variable_name="..."` |
| `.partial()` | Pre-fill variables | Lock in constants early |
| `{variable}` | Dynamic injection | Pass dict to `.invoke()` |

---

← [4. LangChain Basics](04-langchain-basics.md) · [Contents](README.md) · [6. Output Handling](06-output-handling.md) →
