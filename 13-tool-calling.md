# 13. Tool Calling

[Home](README.md) · Part 13 of 14

> Letting the model request actions: function schemas, generated arguments, tool results, and the execution loop.

**In this part:** Tool calling · Function schemas · @tool + Pydantic · Arguments as untrusted input · Tool results · The full loop · Parallel tool calls · Errors & timeouts · Security

---

## 1. Tool Calling (The Model Requests, Your Code Executes)
**What it is:** Tool calling — also called function calling — lets an LLM request that your application run a specific function with specific arguments. The model does **not** execute code itself. It emits a structured request; your code runs the function and returns the result.
- **The flow:**

```text
User: "What's the weather in Paris?"
    ↓
LLM emits: get_weather(city="Paris")
    ↓
Your code calls a weather API
    ↓
Result: {"temp": 22, "condition": "sunny"}
    ↓
LLM generates: "It's 22°C and sunny in Paris."
```

- **Why it exists:** LLMs are bad at real-time data, exact arithmetic, private databases, and side effects. Tools give them access to calculators, search APIs, SQL queries, calendars, and internal services.
- **Tool calling vs agents:** Tool calling is the mechanism. An agent is a loop that repeatedly decides which tool to call next.
- **Why it matters:** Tool calling turns an LLM from a text generator into a system that can fetch facts, perform actions, and interact with real software.

## 2. Function Schemas (The Contract)
**What it is:** A schema describes a tool to the model: its name, purpose, and parameters. It is usually expressed as JSON Schema or generated from Python type hints / Pydantic.
- **Raw JSON Schema example:**

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a city. Use when the user asks about weather.",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "City name, e.g. Paris"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature unit"
        }
      },
      "required": ["city"]
    }
  }
}
```

- **What makes a good schema:**
    - Clear, verb-based name: `get_weather`, `search_orders`, `create_ticket`.
    - Description says *when* to use the tool, not just what it does.
    - Every parameter has a type and description.
    - Use `enum` for fixed choices.
    - Mark required fields explicitly.
- **Why it matters:** The model chooses tools based on names and descriptions. A vague description causes wrong tool selection. A loose schema causes invalid arguments.

## 3. Defining Tools in LangChain (`@tool` + Pydantic)
**What it is:** LangChain lets you define tools as Python functions. The docstring becomes the description; type hints become the schema.
- **Basic tool:**

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city. Use when the user asks about weather."""
    # Mock implementation
    return f"{{\"city\": \"{city}\", \"temp\": 22, \"unit\": \"{unit}\", \"condition\": \"sunny\"}}"
```

- **Strict tool with Pydantic arguments:**

```python
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class WeatherInput(BaseModel):
    city: str = Field(description="City name, e.g. Paris")
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city."""
    return f"{{\"city\": \"{city}\", \"temp\": 22, \"unit\": \"{unit}\"}}"
```

- **Binding tools to a model:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools([get_weather])
```

- **Why it matters:** Pydantic gives you validation, documentation, and type safety. The model gets a precise contract; your code gets predictable inputs.

## 4. Arguments (Model-Generated JSON, Treat as Untrusted)
**What it is:** When the model decides to call a tool, it generates arguments as JSON. These arguments are predictions, not guarantees.
- **What a tool call looks like:**

```python
response = llm_with_tools.invoke("What's the weather in Paris?")

print(response.tool_calls)
# [
#   {
#     "name": "get_weather",
#     "args": {"city": "Paris", "unit": "celsius"},
#     "id": "call_abc123",
#     "type": "tool_call"
#   }
# ]
```

- **Validation pattern:**

```python
from pydantic import ValidationError

for call in response.tool_calls:
    try:
        validated = WeatherInput(**call["args"])
    except ValidationError as e:
        # Return a structured error so the model can recover
        error_result = {
            "error": "invalid_arguments",
            "details": e.errors()
        }
        continue

    result = get_weather.invoke(validated.model_dump())
```

- **Common argument failures:**
    - Missing required field.
    - Wrong type: `"22"` instead of `22`.
    - Hallucinated enum value: `"kelvin"`.
    - Extra unexpected fields.
    - Injected instructions inside a string argument.
- **Security rule:** Never pass model-generated arguments directly into shell commands, raw SQL, or `eval()`. Use allowlists, parameterized queries, and strict validation.
- **Why it matters:** Arguments are the attack surface. A tool that trusts model output blindly can delete data, leak secrets, or execute unintended actions.

## 5. Tool Results (Feeding Evidence Back)
**What it is:** After your code executes a tool, you send the result back to the model as a `ToolMessage`. The model then uses that result to continue reasoning or produce a final answer.
- **LangChain example:**

```python
from langchain_core.messages import ToolMessage
import json

tool_result = {"temp": 22, "unit": "celsius", "condition": "sunny"}

message = ToolMessage(
    content=json.dumps(tool_result),
    tool_call_id="call_abc123"
)
```

- **Raw API shape:**

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temp\": 22, \"condition\": \"sunny\"}"
}
```

- **Good tool results:**
    - Compact JSON, not huge dumps.
    - Include units, IDs, and timestamps.
    - Return errors as structured data, not stack traces.
    - Strip secrets and PII before returning.
- **Why it matters:** The `tool_call_id` links the result to the original request. Without it, the model cannot match results to calls, especially when multiple tools are called in parallel.

## 6. The Full Tool-Calling Loop (Complete Example)
**What it is:** A loop where the model requests tools, your code executes them, results are appended, and the model continues until it produces a final answer.
- **Complete implementation:**

```python
import json
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get current weather for a city. Use when the user asks about weather."""
    return json.dumps({
        "city": city,
        "temp": 22,
        "unit": unit,
        "condition": "sunny"
    })

tools = [get_weather]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

messages = [
    SystemMessage(content="Use tools when they help answer the question."),
    HumanMessage(content="What's the weather in Paris?")
]

for _ in range(5):  # Cap iterations to prevent infinite loops
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        print(ai_message.content)
        break

    for call in ai_message.tool_calls:
        tool_fn = tools_by_name[call["name"]]
        result = tool_fn.invoke(call["args"])

        messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"]
            )
        )
```

- **Why it matters:** This is the core agent loop. Every tool-using agent — customer support bots, coding assistants, data analysts — is built on this pattern.

## 7. Multiple and Parallel Tool Calls
**What it is:** The model can request several tools in one turn. Your code should execute all of them and return a `ToolMessage` for each call ID.
- **Example with two tools:**

```python
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return json.dumps({"city": city, "temp": 22})

@tool
def get_time(city: str) -> str:
    """Get current local time for a city."""
    return json.dumps({"city": city, "time": "14:30"})

llm_with_tools = llm.bind_tools([get_weather, get_time])

response = llm_with_tools.invoke(
    "What's the weather and local time in Tokyo?"
)

# response.tool_calls may contain both get_weather and get_time
for call in response.tool_calls:
    tool_fn = tools_by_name[call["name"]]
    result = tool_fn.invoke(call["args"])
    messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
```

- **Parallel execution:**

```python
import asyncio

async def execute_calls(tool_calls):
    async def run(call):
        tool_fn = tools_by_name[call["name"]]
        result = await tool_fn.ainvoke(call["args"])
        return ToolMessage(content=result, tool_call_id=call["id"])

    return await asyncio.gather(*(run(c) for c in tool_calls))
```

- **Why it matters:** Parallel calls reduce latency. But you must preserve every `tool_call_id`, or the model will receive mismatched results.

## 8. Errors, Validation, and Timeouts
**What it is:** Tools fail. APIs time out, databases are down, arguments are invalid. The model should receive a structured error so it can recover gracefully.
- **Defensive execution:**

```python
import json

def safe_execute(tool_fn, args, tool_call_id):
    try:
        result = tool_fn.invoke(args)
        return ToolMessage(content=result, tool_call_id=tool_call_id)

    except Exception as e:
        # Do not leak stack traces or secrets
        error = {
            "error": "tool_execution_failed",
            "message": "The tool could not complete the request."
        }
        return ToolMessage(
            content=json.dumps(error),
            tool_call_id=tool_call_id
        )
```

- **Failure handling table:**

| Failure | Example | Response |
|---|---|---|
| Invalid args | Missing `city` | Return validation error |
| Tool timeout | Weather API slow | Return timeout error |
| Permission denied | User lacks access | Return authorization error |
| Not found | Order ID missing | Return empty result |
| Unexpected exception | Bug in tool | Log internally, return generic error |

- **Best practices:**
    - Set timeouts on every external call.
    - Retry only transient failures.
    - Cap agent loop iterations.
    - Log tool name, args, result, and latency.
    - Mock tools in tests.
- **Why it matters:** A tool failure should not crash the agent. It should become evidence the model can use to apologize, retry, or ask for clarification.

## 9. Security and Best Practices
**What it is:** Tools give the model indirect access to your systems. That power must be constrained.
- **Security checklist:**
    - Validate every argument with Pydantic.
    - Use least-privilege credentials.
    - Require human approval for destructive actions.
    - Never expose secrets in tool results.
    - Use parameterized queries, not raw SQL strings.
    - Sandbox code execution.
    - Treat retrieved documents as data, not instructions.
    - Allowlist tools per user role.
- **Design guidelines:**

| Practice | Reason |
|---|---|
| Clear tool names and descriptions | Better tool selection |
| Few, focused tools | Less confusion |
| Typed arguments with enums | Fewer invalid calls |
| Compact JSON results | Lower token cost |
| `temperature=0` | More predictable tool use |
| Cap loop iterations | Prevent runaway agents |
| Log every call | Debugging and auditing |
| Human-in-the-loop for writes | Safety |

- **Why it matters:** Tool calling is where LLMs touch reality. Without validation and permissions, a hallucinated argument can become a real-world incident.

## Quick Reference Cheat Sheet

| Concept | Tool / Pattern | Key Syntax |
|---|---|---|
| Define a tool | `@tool` decorator | Docstring + type hints |
| Strict arguments | Pydantic `args_schema` | `@tool(args_schema=MySchema)` |
| Bind tools to LLM | `llm.bind_tools([...])` | Returns tool-aware model |
| Read tool requests | `response.tool_calls` | `name`, `args`, `id` |
| Execute tool | `tool.invoke(args)` | Runs your Python function |
| Return result | `ToolMessage` | `content`, `tool_call_id` |
| Loop until done | Append messages, re-invoke | Cap iterations |
| Parallel calls | `asyncio.gather` | Preserve each call ID |
| Handle errors | Structured error JSON | Return as `ToolMessage` |
| Validate args | Pydantic `ValidationError` | Reject or repair |
| Secure tools | Allowlists, permissions, approval | Never trust model args |

---

← [12. Evaluation and Reliability](12-evaluation-and-reliability.md) · [Contents](README.md) · [14. Agent Fundamentals](14-agent-fundamentals.md) →
