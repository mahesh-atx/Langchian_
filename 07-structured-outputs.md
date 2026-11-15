# 7. Structured Outputs

[Home](README.md) · Part 7 of 14

> Turning free text into typed Python objects with JSON schemas, Pydantic, and `with_structured_output()`.

**In this part:** Why structure matters · JSON responses · Schemas · Pydantic · with_structured_output() · Nested schemas · Validation errors · Best practices

---

## 1. The Problem with Free-Text (Why Structure Matters)
**What it is:** By default, LLMs generate unstructured, free-flowing text. This is great for conversation but nearly useless for software that needs to parse, store, or act on the output programmatically.
- **The Problem:**

```python
# You ask: "Extract the name, age, and city from this text: John is 30 and lives in Paris."
# LLM returns: "Sure! John is 30 years old and he lives in the beautiful city of Paris."
```

Now you have to write fragile regex or string-splitting code to extract the data. If the LLM changes its wording slightly, your parser breaks.
- **The Solution:** Force the LLM to return data in a strict, predictable format (like JSON) that matches a predefined schema.
- **Why it matters:** Structured outputs turn LLMs from "chatbots" into reliable data extraction engines, API backends, and pipeline components.

## 2. JSON Responses (The Universal Format)
**What it is:** JSON (JavaScript Object Notation) is the standard lightweight data format for structured LLM outputs. It uses key-value pairs and arrays that map directly to programming objects.
- **How it works:** You instruct the model (via system prompt, JSON mode, or schema binding) to output *only* valid JSON, with no extra commentary.
- **Example — Manual JSON Prompting:**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

prompt = """
Extract the person's details from the text below.
Respond ONLY with valid JSON. No markdown, no explanations.

Text: "Sarah Connor, age 45, resides in Los Angeles."

JSON format:
{"name": "...", "age": ..., "city": "..."}
"""

response = llm.invoke(prompt)
print(response.content)
# {"name": "Sarah Connor", "age": 45, "city": "Los Angeles"}
```

- **The Risk:** Even with instructions, the model might:
    - Wrap the JSON in markdown code fences (markdown code fences)
    - Add a preamble like "Here is the JSON:"
    - Miss a comma or bracket
- **Why it matters:** Manual prompting works for simple cases but is fragile. For production, you need schema enforcement.

## 3. Schemas (The Blueprint)
**What it is:** A schema is a strict definition of what the output *must* look like. It specifies the field names, data types, and constraints. Think of it as a contract between you and the LLM.
- **How it works:** You define the schema, pass it to the model, and the model is constrained to generate output that matches it exactly. The API provider (OpenAI, Anthropic, etc.) enforces this at the token-generation level.
- **Schema Definition (JSON Schema format):**

```json
{
  "type": "object",
  "properties": {
    "name": { "type": "string" },
    "age": { "type": "integer" },
    "city": { "type": "string" },
    "is_active": { "type": "boolean" }
  },
  "required": ["name", "age", "city"]
}
```

- **Why it matters:** Schemas eliminate ambiguity. The model *cannot* return a string where an integer is expected, or omit a required field. This makes downstream parsing 100% reliable.

## 4. Pydantic (The Python Validation Layer)
**What it is:** Pydantic is a Python library that defines data models as classes with type hints. It automatically validates, serializes, and documents data. It is the standard way to define schemas for LLM structured outputs in LangChain.
- **How it works:** You define a Python class inheriting from `BaseModel`. Each field has a name and a type. Pydantic handles validation automatically.
- **Basic Example:**

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """Represents a person extracted from text."""
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    city: str = Field(description="The city where they live")
    is_active: bool = Field(default=True, description="Whether they are currently active")
```

- **What Pydantic Gives You:**
    - **Type enforcement:** If the LLM returns `"thirty"` instead of `30`, Pydantic raises an error.
    - **Documentation:** The `description` fields are passed to the LLM as hints, improving accuracy.
    - **Defaults:** Optional fields can have fallback values.
    - **Validation:** Automatic checking of constraints (min/max values, string lengths, etc.)
- **Why it matters:** Pydantic turns a vague "give me some data" request into a strict, self-documenting, validated contract. It's the bridge between LLM text generation and reliable Python objects.

## 5. LangChain `with_structured_output()` (The Magic Method)
**What it is:** A LangChain method that binds a Pydantic schema (or JSON Schema) to a chat model, forcing the LLM to return data that exactly matches your schema.
- **How it works:** Under the hood, it uses the provider's native structured output feature (like OpenAI's `response_format` or Anthropic's tool-use) to constrain token generation.
- **Complete Example:**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Step 1: Define the schema
class MovieReview(BaseModel):
    """Structured output for a movie review."""
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    summary: str = Field(description="A one-sentence summary")
    recommended: bool = Field(description="Whether to recommend it")

# Step 2: Create the model and bind the schema
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm = llm.with_structured_output(MovieReview)

# Step 3: Invoke — returns a Pydantic object, NOT a string
result = structured_llm.invoke("Review the movie Inception (2010).")

# Step 4: Access fields directly as Python attributes
print(result.title)         # "Inception"
print(result.rating)        # 8.8
print(result.summary)       # "A mind-bending heist within dreams."
print(result.recommended)   # True
print(type(result))         # <class 'MovieReview'>
```

- **Why it matters:** You skip all manual JSON parsing, regex, and error-prone string manipulation. You get a validated Python object directly.

## 6. Nested and Complex Schemas (Real-World Structures)
**What it is:** Schemas that contain lists, nested objects, enums, and optional fields to represent complex real-world data.
- **How it works:** Pydantic supports nested `BaseModel` classes, `List[]`, `Optional[]`, and `Enum` types. LangChain passes the full nested schema to the LLM.
- **Example — Extracting a Full Order:**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class OrderItem(BaseModel):
    product_name: str = Field(description="Name of the product")
    quantity: int = Field(description="Number of units", ge=1)
    unit_price: float = Field(description="Price per unit")

class CustomerOrder(BaseModel):
    """A complete customer order extracted from an email."""
    customer_name: str
    email: str
    items: List[OrderItem] = Field(description="List of ordered items")
    priority: Priority = Field(default=Priority.medium)
    special_instructions: Optional[str] = Field(default=None)

# Bind and invoke
structured_llm = llm.with_structured_output(CustomerOrder)

result = structured_llm.invoke("""
    Extract order from this email:
    "Hi, I'm Jane Smith (jane@example.com). I'd like 2 laptops at $999 each
    and 1 mouse at $25. Please rush this order. Thanks!"
""")

print(result.customer_name)           # "Jane Smith"
print(result.items[0].product_name)   # "laptops"
print(result.items[0].quantity)       # 2
print(result.priority)                # Priority.high
print(result.special_instructions)    # "Please rush this order."
```

- **Why it matters:** Real data is rarely flat. Nested schemas let you extract entire relational structures (orders with line items, articles with sections, code with functions) in a single LLM call.

## 7. Handling Validation Errors (When Things Go Wrong)
**What it is:** Even with structured outputs, things can fail. The model might hit a token limit, the provider might have a transient error, or the output might not perfectly match the schema.
- **Common Errors:**
    - `ValidationError`: Pydantic rejects the output (wrong type, missing field).
    - `OutputParserException`: LangChain can't parse the response.
    - Truncation: The model runs out of tokens mid-JSON.
- **Defensive Pattern:**

```python
from pydantic import ValidationError

def safe_extract(llm, schema, prompt, max_retries=3):
    structured_llm = llm.with_structured_output(schema)

    for attempt in range(max_retries):
        try:
            result = structured_llm.invoke(prompt)
            return result  # Success!
        except ValidationError as e:
            print(f"Attempt {attempt+1}: Validation failed - {e}")
            # Optionally retry with a stricter prompt
        except Exception as e:
            print(f"Attempt {attempt+1}: Unexpected error - {e}")

    return None  # All retries exhausted

# Usage
order = safe_extract(llm, CustomerOrder, "Extract order from: ...")
if order is None:
    print("Failed to extract order after retries.")
```

- **Why it matters:** In production, you cannot assume every LLM call succeeds. Retry logic with validation ensures your pipeline degrades gracefully instead of crashing.

## 8. Best Practices (Production Tips)
**What it is:** Guidelines to make structured outputs reliable, fast, and cost-effective.
- **Key Principles:**

| Practice | Reason |
|---|---|
| Always add `Field(description=...)` | Helps the LLM understand what each field means |
| Use `Enum` for fixed choices | Prevents the model from inventing new categories |
| Use `Optional[]` for non-critical fields | Avoids failures when data is missing |
| Set `temperature=0` | Reduces randomness, improves schema compliance |
| Keep schemas focused | Don't ask for 50 fields in one call; split into multiple |
| Validate with Pydantic *after* extraction | Double-check business logic (e.g., `age > 0`) |
| Use `include_raw=True` for debugging | Get both the parsed object AND the raw response |

- **Debugging with `include_raw`:**

```python
structured_llm = llm.with_structured_output(MovieReview, include_raw=True)

result = structured_llm.invoke("Review The Matrix.")

print(result["raw"].content)     # The raw LLM text
print(result["parsed"].title)    # The parsed Pydantic object
print(result["parsing_error"])   # None if successful, error message if failed
```

- **Why it matters:** Structured outputs are powerful but not magic. Following these practices ensures your LLM pipeline is deterministic, debuggable, and production-ready.

## Quick Reference Cheat Sheet

| Concept | Tool | Key Syntax |
|---|---|---|
| Define schema | Pydantic `BaseModel` | `class Foo(BaseModel):` |
| Add descriptions | Pydantic `Field` | `Field(description="...")` |
| Bind schema to LLM | LangChain | `llm.with_structured_output(Schema)` |
| Get result | Invoke | `result = structured_llm.invoke("...")` |
| Access fields | Python attributes | `result.field_name` |
| Nested objects | `List[SubModel]` | `items: List[OrderItem]` |
| Fixed choices | `Enum` | `class Color(str, Enum):` |
| Optional fields | `Optional[str]` | `nickname: Optional[str] = None` |
| Debug raw output | `include_raw=True` | `result["raw"]` |
| Handle failures | `try/except ValidationError` | Retry loop |

---

← [6. Output Handling](06-output-handling.md) · [Contents](README.md) · [8. Conversation History](08-conversation-history.md) →
