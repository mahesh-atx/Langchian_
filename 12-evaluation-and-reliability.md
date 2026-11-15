# 12. Evaluation and Reliability

[Home](README.md) · Part 12 of 14

> Testing probabilistic software: golden datasets, semantic and LLM-judge scoring, groundedness, retries, and guardrails.

**In this part:** Why evaluate · Quality dimensions · Golden datasets · Deterministic / semantic / judge tests · RAG evaluation · Failures & retries · Reducing hallucinations · Production reliability

---

## 1. Why Evaluation Matters (Probabilistic Software)
**What it is:** LLM applications are not deterministic. The same prompt can produce different answers across runs, model versions, temperatures, or retrieved context. Evaluation is how you measure whether your system is actually working — and keep it working after changes.
- **The core problem:**

```text
Traditional software: same input → same output → unit test passes/fails
LLM software:         same input → different wording → is it still correct?
```

- **What can silently break:**
    - A model update changes behavior.
    - A prompt tweak improves one case but breaks ten others.
    - Retrieval starts returning worse chunks after re-indexing.
    - A schema change causes parsing failures.
    - Costs or latency spike without obvious errors.
- **Why it matters:** Without evaluation, you are shipping vibes. Reliability comes from measurable tests, failure handling, and continuous monitoring — not from hoping the model behaves.

## 2. What to Test (Quality Dimensions)
**What it is:** “Good answer” is multidimensional. You need separate checks for correctness, format, grounding, safety, and operational behavior.
- **Key dimensions:**

| Dimension | Question | Example Check |
|---|---|---|
| Correctness | Is the answer factually right? | Expected facts present |
| Format compliance | Does it match the required schema? | Pydantic validation |
| Groundedness | Is it supported by retrieved context? | No unsupported claims |
| Relevance | Does it answer the actual question? | Rubric or judge score |
| Completeness | Are all required parts included? | Required fields/citations |
| Safety | No harmful, biased, or leaked data? | Policy checks, PII filters |
| Robustness | Handles edge cases and bad input? | Empty docs, adversarial prompts |
| Operations | Latency, cost, error rate acceptable? | p95 latency, token budget |

- **Rule of thumb:** Test the cheap deterministic things first: schema, required fields, citations, forbidden phrases. Use semantic or LLM-based evaluation only where necessary.
- **Why it matters:** A beautiful, fluent answer can still be wrong, ungrounded, unsafe, or too expensive. Evaluate the dimensions that matter for your use case.

## 3. Golden Datasets (Your Regression Suite)
**What it is:** A curated set of representative test cases with expected behavior. This is your LLM application’s regression suite.
- **What to include:**
    - Common real user questions.
    - Known hard cases.
    - Questions with exact facts, numbers, dates, or IDs.
    - Unanswerable questions where the correct response is “I don’t know.”
    - Out-of-scope or adversarial inputs.
    - Multi-turn conversations if history matters.
    - Cases that previously failed in production.
- **Example:**

```python
golden_cases = [
    {
        "question": "What is the refund window?",
        "expected_facts": ["30 days"],
        "must_not_contain": ["60 days", "90 days"],
        "should_answer": True,
    },
    {
        "question": "What is the CEO's personal phone number?",
        "expected_facts": [],
        "must_not_contain": ["+1", "555"],
        "should_answer": False,
    },
    {
        "question": "Summarize the parental leave policy.",
        "expected_facts": ["12 weeks", "paid"],
        "must_not_contain": [],
        "should_answer": True,
    },
]
```

- **Why it matters:** Every prompt change, model upgrade, or retrieval change should be run against this suite. If pass rate drops, you caught a regression before users did.

## 4. Testing Answers (Deterministic, Semantic, and Judge-Based)
**What it is:** Three layers of answer testing, from cheap and strict to expensive and nuanced.
- **Layer 1: Deterministic checks**
Use for stable facts, formats, and required fields.

```python
from pydantic import BaseModel

class RefundPolicy(BaseModel):
    window_days: int
    conditions: str

structured_llm = llm.with_structured_output(RefundPolicy)
result = structured_llm.invoke("Extract the refund policy from the context.")

assert result.window_days == 30
assert "receipt" in result.conditions.lower()
```

- **Layer 2: Semantic similarity**
Use when wording can vary but meaning should match.

```python
import numpy as np
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_match(answer: str, expected: str, threshold: float = 0.82):
    va = embeddings.embed_query(answer)
    ve = embeddings.embed_query(expected)
    return cosine_similarity(va, ve) >= threshold

assert semantic_match(answer, "Refunds are available within 30 days.")
```

**Warning:** Similarity is not truth. Tune thresholds on real examples.
- **Layer 3: LLM-as-judge**
Use a model to grade another model’s answer against a rubric.

```python
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

class Verdict(BaseModel):
    passed: bool
    score: int = Field(ge=1, le=5)
    reason: str
    unsupported_claims: List[str] = []

judge_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict evaluator. Grade the answer using the rubric.\n"
     "5 = fully correct, grounded, complete, and relevant.\n"
     "3 = partially correct or missing minor details.\n"
     "1 = wrong, hallucinated, or irrelevant.\n"
     "Return structured output only."),
    ("human",
     "Question: {question}\n"
     "Expected facts: {expected_facts}\n"
     "Context: {context}\n"
     "Answer: {answer}")
])

judge = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(Verdict)
judge_chain = judge_prompt | judge

verdict = judge_chain.invoke({
    "question": "What is the refund window?",
    "expected_facts": "30 days",
    "context": retrieved_context,
    "answer": answer,
})

assert verdict.passed
assert verdict.score >= 4
```

- **Judge limitations:** LLM judges can be biased toward longer answers, their own style, or confident wording. Calibrate against human labels, use clear rubrics, and consider pairwise comparisons for important decisions.
- **Why it matters:** Deterministic tests catch obvious breakage. Semantic tests catch meaning drift. Judges catch nuanced quality issues. Use all three where appropriate.

## 5. RAG-Specific Evaluation (Groundedness First)
**What it is:** In RAG systems, you must evaluate retrieval and generation separately. A wrong answer may be caused by missing context, not a bad model.
- **Core RAG metrics:**

| Metric | Meaning | Failure Example |
|---|---|---|
| Context recall | Did retrieval find the needed information? | Policy exists but chunk not retrieved |
| Context precision | Are retrieved chunks relevant? | Top-k full of unrelated text |
| Faithfulness / groundedness | Is the answer supported by context? | Model invents a 60-day refund |
| Answer relevance | Does the answer address the question? | Answers about shipping instead of refunds |
| Citation correctness | Do cited sources actually support the claim? | Citation points to wrong page |

- **Debugging order:**

```text
1. Inspect retrieved chunks.
2. Check whether the answer exists in those chunks.
3. Check whether the model used the chunks correctly.
4. Only then blame the LLM.
```

- **Faithfulness check example:**

```python
class Faithfulness(BaseModel):
    supported: bool
    unsupported_claims: List[str]

verifier_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Check whether every claim in the answer is supported by the context. "
     "List any unsupported claims."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}")
])

verifier = verifier_prompt | llm.with_structured_output(Faithfulness)

check = verifier.invoke({
    "context": retrieved_context,
    "answer": answer,
})

if not check.supported:
    print("Unsupported claims:", check.unsupported_claims)
```

- **Useful tools:** LangSmith, RAGAS, DeepEval, promptfoo, and custom pytest suites can automate parts of this.
- **Why it matters:** Most RAG failures are retrieval failures. If the right evidence never reaches the model, no amount of prompt engineering will reliably fix the answer.

## 6. Handling Failures (Retries, Fallbacks, Timeouts)
**What it is:** Reliability means expecting failure and designing recovery paths. LLM apps fail through network errors, rate limits, truncation, invalid output, refusals, and bad retrieval.
- **Failure categories:**

| Failure Type | Example | Response |
|---|---|---|
| Transient | Timeout, 429 rate limit | Retry with backoff |
| Permanent | Bad API key, invalid request | Fail fast, alert |
| Output failure | Invalid JSON, missing field | Validate, retry, repair |
| Quality failure | Failed eval, unsupported claim | Regenerate, fallback, escalate |
| Safety failure | PII leak, harmful content | Block, redact, escalate |
| Retrieval failure | No relevant docs | Say “I don’t know” |

- **LangChain retry + fallback pattern:**

```python
from langchain_openai import ChatOpenAI

primary = ChatOpenAI(model="gpt-4o", temperature=0, timeout=30)
fallback = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=30)

reliable_llm = (
    primary
    .with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True
    )
    .with_fallbacks([fallback])
)

response = reliable_llm.invoke("Answer the user question.")
```

- **Manual guardrail pattern:**

```python
import time
import random

def reliable_invoke(chain, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            answer = chain.invoke(question)

            if not answer or len(answer.strip()) < 5:
                raise ValueError("Empty or too-short answer")

            return answer

        except Exception as e:
            if attempt == max_retries - 1:
                return "I’m unable to answer reliably right now."

            sleep = (2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep)
```

- **Important:** Do not blindly retry permanent errors like authentication failures or invalid schemas. Classify errors first.
- **Why it matters:** A single timeout should not crash your app. A single malformed output should not corrupt your database. Reliability is built from validation, retries, fallbacks, and graceful degradation.

## 7. Reducing Hallucinations (Grounding + Verification)
**What it is:** Hallucinations are confident but unsupported or false outputs. You reduce them by constraining the model, grounding it in evidence, and verifying claims.
- **Practical techniques:**

| Technique | How it helps |
|---|---|
| Lower temperature | Reduces random variation |
| RAG with good retrieval | Provides factual context |
| “Answer only from context” | Limits invention |
| “Say I don’t know if unsure” | Gives model an escape hatch |
| Require citations | Makes claims traceable |
| Structured outputs | Reduces vague free-text drift |
| Tool use for math/current facts | Avoids mental arithmetic and stale knowledge |
| Verification pass | Checks claims against evidence |
| Metadata filters | Prevents wrong-source answers |
| Reranking | Improves context quality |

- **Grounded prompt pattern:**

```python
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using only the provided context. "
     "Every factual claim must be supported by the context. "
     "If the context does not contain the answer, say: "
     "'I don't have enough information to answer that.' "
     "Cite sources using [source:page].\n\n"
     "Context:\n{context}"),
    ("human", "{question}")
])
```

- **Verify citations exist:**

```python
def citations_are_valid(answer: str, docs: list) -> bool:
    allowed_sources = {
        f"{d.metadata.get('source')}:{d.metadata.get('page')}"
        for d in docs
    }

    # Simplified example: extract citations and check against allowed sources
    # In production, parse citations robustly.
    return all(
        citation in allowed_sources
        for citation in extract_citations(answer)
    )
```

- **Do not rely only on self-checking:** Asking the same model “are you sure?” is weak verification. Prefer external evidence, retrieval checks, tools, or a separate verifier.
- **Why it matters:** Hallucinations cannot be eliminated entirely, but they can be made rare, detectable, and recoverable. The goal is not perfection — it is controlled risk.

## 8. Production Reliability (Monitoring, Versioning, Guardrails)
**What it is:** Evaluation does not stop at deployment. You need observability, version control, and feedback loops to keep the system reliable over time.
- **What to log:**
    - Request ID and session ID.
    - Model name and version.
    - Prompt template version.
    - Retrieved document IDs and scores.
    - Token usage and cost.
    - Latency and finish reason.
    - Validation errors and retries.
    - Eval scores or judge verdicts.
    - User feedback and escalations.
- **What to monitor:**

| Metric | Why it matters |
|---|---|
| Error rate | Detects API or parsing failures |
| p95 latency | Detects slow retrievals or long generations |
| Cost per request | Detects runaway context or retries |
| Eval pass rate | Detects quality regression |
| “I don’t know” rate | Detects retrieval coverage issues |
| Unsupported claim rate | Detects hallucination drift |
| User thumbs-down rate | Detects real-world dissatisfaction |

- **Versioning discipline:**

```text
Pin model versions where possible.
Version prompts like code.
Run golden-set evals before deploying prompt changes.
Canary new models on a small traffic slice.
Keep rollback paths.
```

- **Guardrails checklist:**
    - Validate structured outputs before saving.
    - Enforce token and cost budgets.
    - Filter PII and secrets.
    - Apply retrieval permission filters.
    - Treat retrieved documents as data, not instructions.
    - Escalate to humans for high-risk decisions.
- **Why it matters:** LLM systems drift. Models change, documents change, users change. Reliability is a continuous process: test, deploy, monitor, learn, and retest.

## Quick Reference Cheat Sheet

| Goal | Technique | Key Tool / Pattern |
|---|---|---|
| Catch format errors | Schema validation | Pydantic, `with_structured_output()` |
| Catch fact errors | Golden dataset | Expected facts, assertions |
| Catch meaning drift | Semantic similarity | Embeddings + cosine threshold |
| Grade nuanced quality | LLM-as-judge | Rubric + structured verdict |
| Evaluate RAG | Faithfulness + context recall | Inspect retrieval first |
| Handle transient errors | Retry with backoff | `.with_retry()` |
| Handle provider failure | Fallback model | `.with_fallbacks()` |
| Prevent hangs | Timeouts | `timeout=30` |
| Reduce hallucination | Grounding + citations | “Answer only from context” |
| Detect unsupported claims | Verifier pass | Separate faithfulness check |
| Monitor production | Logs + metrics | LangSmith, tracing, dashboards |
| Prevent regressions | CI eval suite | Run golden cases on every change |

---

← [11. RAG (Retrieval-Augmented Generation)](11-rag-retrievalaugmented-generation.md) · [Contents](README.md) · [13. Tool Calling](13-tool-calling.md) →
