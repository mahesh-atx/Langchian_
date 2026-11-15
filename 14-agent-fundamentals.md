# 14. Agent Fundamentals

[Home](README.md) · Part 14 of 14

> LLM + tools + loop: the agent cycle, tool selection, stopping conditions, state, and production patterns.

**In this part:** What an agent is · The agent loop · Tool selection · Stopping conditions · State & memory · Manual loop example · LangGraph & prebuilt agents · Failure modes · Evaluation

---

## 1. What an Agent Is (LLM + Tools + Loop)
**What it is:** An agent is an LLM placed inside a control loop where it can decide which tools to call, observe the results, and decide what to do next — until it produces a final answer or hits a stopping condition.
- **The key distinction:**

```text
Chain:  You hardcode the steps.        prompt → retrieve → llm → answer
Router: One decision, then fixed path. classify → branch A or B
Agent:  The model chooses the path.    decide → act → observe → decide → ...
```

- **Core components:**
    - **Model:** the decision function.
    - **Tools:** the actions it can request.
    - **State:** messages, tool results, scratchpad, task metadata.
    - **Loop:** the code that executes tools and feeds results back.
    - **Policy:** stopping rules, budgets, permissions, approval gates.
- **Rule of thumb:** If you can write the steps yourself, use a chain. Use an agent only when the path is genuinely unknown at design time.
- **Why it matters:** Agents are not magic autonomy. They are a `while` loop with an LLM as the planner. Reliability comes from the loop’s guardrails, not from the model’s confidence.

## 2. The Agent Loop (Observe → Decide → Act → Repeat)
**What it is:** The repeating cycle that makes an agent different from a single LLM call. Often called the ReAct pattern: reason, act, observe.
- **The loop:**

```text
1. Observe:  Read task + history + tool results
2. Decide:   Answer directly OR request tool calls
3. Act:      Your code executes the requested tools
4. Observe:  Append tool results to state
5. Repeat:   Until final answer or stop condition
```

- **State grows each turn:**

```python
messages = [
    SystemMessage(content="You are a careful research agent."),
    HumanMessage(content="Compare our refund policy with our shipping policy."),
    # AIMessage with tool_calls: search_docs("refund policy")
    # ToolMessage: refund policy text
    # AIMessage with tool_calls: search_docs("shipping policy")
    # ToolMessage: shipping policy text
    # AIMessage: final comparison answer
]
```

- **Important:** The model does not execute tools. It emits structured requests. Your application is the executor and the gatekeeper.
- **Why it matters:** Every agent failure — loops, thrashing, runaway cost — is a loop-control failure. Design the loop before you design the prompt.

## 3. Tool Selection (How the Model Chooses)
**What it is:** The model picks tools based on names, descriptions, argument schemas, and the current conversation. Selection is probabilistic, so it must be engineered.
- **What drives selection:**

| Factor | Effect |
|---|---|
| Tool name | `search_orders` beats `do_stuff` |
| Description | Should say *when* to use it |
| Argument schema | Typed, enum-constrained args reduce errors |
| System prompt | Can encode routing policy |
| Tool count | Too many tools degrades selection |
| Prior results | Failed calls should change strategy |

- **Good vs bad descriptions:**

```python
# Bad: vague
@tool
def lookup(q: str) -> str:
    """Look things up."""

# Good: specific trigger conditions
@tool
def search_policy_docs(query: str) -> str:
    """Search internal policy documents. Use for questions about refunds,
    shipping, returns, or employee policies. Do not use for general knowledge."""
```

- **Selection strategies:**
    - **Auto:** let the model choose (default).
    - **Forced:** some providers support `tool_choice` to require a specific tool.
    - **Dynamic subsets:** expose only tools relevant to the current task or user role.
    - **Tool retrieval:** for large catalogs, embed tool descriptions and retrieve the top-k tools before binding.
    - **Specialized agents:** split into researcher, coder, and reviewer agents instead of one agent with 40 tools.
- **Why it matters:** Wrong tool selection is one of the most common agent failures. Fewer, clearer tools usually beat a giant toolbox.

## 4. Stopping Conditions (When to Quit)
**What it is:** Explicit rules that terminate the loop. Without them, agents can loop forever, burn budget, or thrash on the same failing tool.
- **Stopping condition table:**

| Condition | Type | Why |
|---|---|---|
| No tool calls in response | Natural stop | Model produced final answer |
| Max steps reached | Hard stop | Prevents infinite loops |
| Wall-clock deadline | Hard stop | Protects latency SLA |
| Token/cost budget exceeded | Hard stop | Protects spend |
| Repeated identical call | Heuristic | Detects thrashing |
| No progress for N steps | Heuristic | Detects stalling |
| Error budget exceeded | Hard stop | Stops broken tools |
| Human approval required | Pause | Safety for destructive actions |
| Explicit `finish` tool | Natural stop | Forces structured termination |

- **Natural stop example:**

```python
ai_message = llm_with_tools.invoke(messages)

if not ai_message.tool_calls:
    return ai_message.content  # Done
```

- **Hard stop example:**

```python
MAX_STEPS = 8
DEADLINE_SECONDS = 60

for step in range(MAX_STEPS):
    if time.time() - start > DEADLINE_SECONDS:
        return "Stopped: time budget exceeded."
    ...
return "Stopped: max steps reached."
```

- **Anti-thrash example:**

```python
key = (call["name"], json.dumps(call["args"], sort_keys=True))

if key in seen_calls:
    result = json.dumps({
        "error": "repeated_call",
        "hint": "This exact call already ran. Try a different approach or answer."
    })
else:
    seen_calls.add(key)
    result = tool_fn.invoke(call["args"])
```

- **Why it matters:** “The model will know when to stop” is not a reliability strategy. Stopping conditions are the difference between a demo and a production agent.

## 5. State and Memory (What the Agent Carries)
**What it is:** The accumulated context the agent uses to make decisions: messages, tool results, task state, and checkpoints.
- **Types of state:**
    - **Message history:** system, human, AI, tool messages.
    - **Scratchpad:** notes, plans, intermediate findings.
    - **Task state:** goal, constraints, progress flags.
    - **Checkpoints:** saved state for resume, retry, or audit.
- **Context management:**

```python
from langchain_core.messages import trim_messages

# Keep state within token budget
messages = trim_messages(
    messages,
    max_tokens=6000,
    strategy="last",
    token_counter=llm,
    include_system=True
)
```

- **Checkpointing:** In production, persist state after each step so a crash does not lose progress and a human can inspect the trajectory.
- **Why it matters:** Agents fail when state grows unbounded or when tool results are lost. Trim carefully, preserve tool-call linkage, and checkpoint long tasks.

## 6. Manual Agent Loop in LangChain (Complete Example)
**What it is:** A full agent loop with tool execution, validation, anti-thrash detection, and stopping conditions.
- **Implementation:**

```python
import json
import time
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

@tool
def search_policy_docs(query: str) -> str:
    """Search internal policy documents for refunds, shipping, or returns."""
    return json.dumps({"query": query, "result": "Refunds allowed within 30 days."})

@tool
def add_numbers(a: float, b: float) -> str:
    """Add two numbers exactly. Use for arithmetic instead of mental math."""
    return json.dumps({"sum": a + b})

tools = [search_policy_docs, add_numbers]
tools_by_name = {t.name: t for t in tools}

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

def run_agent(question: str, max_steps: int = 6, timeout_s: int = 60):
    messages = [
        SystemMessage(content=(
            "You are a careful agent. Use tools when they help. "
            "When you have enough information, answer directly without tool calls."
        )),
        HumanMessage(content=question)
    ]

    start = time.time()
    seen_calls = set()

    for step in range(max_steps):
        if time.time() - start > timeout_s:
            return "Stopped: time budget exceeded."

        ai_message = llm_with_tools.invoke(messages)
        messages.append(ai_message)

        # Natural stop
        if not ai_message.tool_calls:
            return ai_message.content

        for call in ai_message.tool_calls:
            # Unknown tool guard
            if call["name"] not in tools_by_name:
                messages.append(ToolMessage(
                    content=json.dumps({"error": "unknown_tool"}),
                    tool_call_id=call["id"]
                ))
                continue

            # Anti-thrash guard
            key = (call["name"], json.dumps(call["args"], sort_keys=True))
            if key in seen_calls:
                messages.append(ToolMessage(
                    content=json.dumps({
                        "error": "repeated_call",
                        "hint": "Try a different approach or answer."
                    }),
                    tool_call_id=call["id"]
                ))
                continue

            seen_calls.add(key)

            # Safe execution
            try:
                result = tools_by_name[call["name"]].invoke(call["args"])
            except Exception:
                result = json.dumps({"error": "tool_execution_failed"})

            messages.append(ToolMessage(
                content=result,
                tool_call_id=call["id"]
            ))

    return "Stopped: max steps reached."

print(run_agent("What is the refund window, and what is 17.5 + 22.25?"))
```

- **Why it matters:** This is the minimal production-shaped agent. Every feature — caps, dedupe, error handling, unknown-tool guards — exists because agents fail in these exact ways.

## 7. Prebuilt Agents and LangGraph (Production Pattern)
**What it is:** LangGraph is the recommended way to build stateful, durable agents. It provides graph-based control flow, checkpointing, and prebuilt ReAct agents.
- **Prebuilt agent example:**

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

agent = create_react_agent(
    llm,
    tools,
    checkpointer=MemorySaver()
)

result = agent.invoke(
    {"messages": [HumanMessage(content="What is the refund policy?")]},
    config={
        "configurable": {"thread_id": "user-123"},
        "recursion_limit": 10
    }
)

print(result["messages"][-1].content)
```

- **What LangGraph adds:**
    - Explicit nodes and edges for control flow.
    - Checkpointing and resume.
    - Human-in-the-loop interrupts.
    - Streaming of intermediate steps.
    - Clear place to enforce budgets and approvals.
- **Why it matters:** Manual loops are great for learning. Graph-based agents are easier to debug, persist, and operate in production.

## 8. Common Failure Modes and Fixes
**What it is:** Agents fail in predictable patterns. Knowing them speeds up debugging.
- **Failure table:**

| Symptom | Cause | Fix |
|---|---|---|
| Infinite loop | No stopping condition | Max steps + deadline |
| Same tool called repeatedly | No progress detection | Dedupe calls, add hint |
| Hallucinated tool name | Too many tools, vague names | Allowlist, fewer tools |
| Wrong tool chosen | Poor descriptions | Rewrite descriptions, add examples |
| Premature final answer | Weak system prompt | Require evidence before answering |
| Cost blowup | Unbounded context/steps | Token budget, trim, caps |
| Tool result ignored | State mismanagement | Verify ToolMessage linkage |
| Unsafe action | No approval gate | Human-in-the-loop for writes |
| Prompt injection via tool result | Untrusted content | Treat tool output as data |

- **Debugging tip:** Log the full trajectory — every tool name, args, result, and step count. Most agent bugs are visible in the trajectory.
- **Why it matters:** Agent reliability is mostly loop engineering. The model is the planner; your code is the safety system.

## 9. Best Practices and Evaluation
**What it is:** Guidelines for building agents that are useful, safe, and measurable.
- **Design principles:**
    - Start with a chain. Add an agent only when necessary.
    - Keep the tool count small and focused.
    - Write descriptions that say *when* to use each tool.
    - Use typed, validated arguments.
    - Set `temperature=0` for predictable decisions.
    - Cap steps, time, and cost.
    - Require approval for destructive actions.
    - Checkpoint long-running tasks.
    - Mock tools in tests.
- **What to evaluate:**

| Metric | Meaning |
|---|---|
| Task success rate | Did it achieve the goal? |
| Tool selection accuracy | Right tool, right args? |
| Steps per task | Efficiency |
| Cost per task | Budget control |
| Latency p95 | User experience |
| Unsafe action rate | Safety |
| Human escalation rate | Autonomy boundary |
| Loop/timeout rate | Stability |

- **Why it matters:** Agents are powerful but expensive and risky. Measure success, efficiency, and safety together — not just whether the final answer looks good.

## Quick Reference Cheat Sheet

| Concept | Pattern / Tool | Key Idea |
|---|---|---|
| Agent | LLM + tools + loop | Model decides path |
| Chain vs agent | Fixed steps vs dynamic loop | Use chains when possible |
| Agent loop | Observe → decide → act → repeat | ReAct pattern |
| Tool selection | Names, descriptions, schemas | Fewer, clearer tools |
| Forced tool | `tool_choice` (provider-specific) | Use sparingly |
| Natural stop | `if not ai_message.tool_calls` | Final answer produced |
| Hard stop | Max steps, deadline, budget | Prevent runaway loops |
| Anti-thrash | Dedupe identical calls | Detect no progress |
| State | Messages + tool results + scratchpad | Trim and checkpoint |
| Prebuilt agent | `create_react_agent` | LangGraph production pattern |
| Safety | Allowlists, validation, approval | Never trust model args |
| Evaluation | Success, steps, cost, safety | Measure the whole trajectory |

---

← [13. Tool Calling](13-tool-calling.md) · [Contents](README.md)
