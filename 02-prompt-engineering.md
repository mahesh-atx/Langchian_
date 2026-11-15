# 2. Prompt Engineering

[Home](README.md) · Part 2 of 14

> Who says what: system prompts set the rules, user prompts set the task, and few-shot examples show the pattern.

**In this part:** System prompts · User prompts · Few-shot prompting

---

## 1. System Prompts (The Director's Rulebook)
**What it is:** The foundational, often hidden instructions given to the AI *before* the user even types their first message. It sets the AI's persona, rules, constraints, and overarching tone.
- **How it works:** In an API or application, the system prompt sits at the very top of the conversation context in a special `system` role. The model reads this first and treats it as the "ground truth" or the rules of the game for the entire session.
- **Examples:**
    - *Persona:* "You are an expert senior software engineer. You write clean, heavily commented Python code."
    - *Constraints:* "Never discuss politics or religion. If asked, politely decline."
    - *Formatting:* "Always output your final answer in strict JSON format."
- **Why it matters:** It prevents the AI from drifting off-topic, ensures consistency across thousands of user interactions, and acts as a primary defense mechanism to enforce safety and specific brand voices.

## 2. User Prompts (The Actor's Cue)
**What it is:** The specific questions, commands, or inputs provided by the end-user. This is the immediate trigger that asks the AI to perform a task.
- **How it works:** The AI combines the System Prompt + Conversation History + the current User Prompt to generate the next tokens.
- **Anatomy of a Great User Prompt:** While you *can* just type "Write a blog," engineered prompts usually contain four elements:
    1.  **Context:** Background information ("I am writing for a beginner gardening blog...")
    2.  **Task:** The specific action ("...write an article about growing tomatoes.")
    3.  **Instructions:** Specific rules ("...include a section on common pests, and keep the tone encouraging.")
    4.  **Output Format:** How it should look ("...Format it with Markdown headers and a bulleted summary at the end.")
- **Examples:** "Summarize the attached financial report into three bullet points, focusing only on Q3 revenue, and output it as a table."
- **Why it matters:** LLMs suffer from "garbage in, garbage out." A vague user prompt yields a generic, hallucination-prone response. A highly specific user prompt acts as a scalpel, yielding precise, highly useful results.

## 3. Few-Shot Prompting (Learning by Example)
**What it is:** A powerful technique where you provide the AI with a few examples (the "shots") of the desired input-output pairs within your prompt, *before* asking it to do the actual task.
- **How it works:** LLMs are fundamentally advanced pattern-matching engines. Instead of trying to explain a complex formatting rule or classification logic with a long paragraph of text, you simply *show* the model what you want. It instantly recognizes the pattern and mimics it.
- **The "Shot" Spectrum:**
    - **Zero-Shot:** Asking the model to do something with *no* examples. (e.g., "Classify this tweet's sentiment: 'The food was okay.'")
    - **One-Shot:** Providing exactly *one* example before the task.
    - **Few-Shot:** Providing *two to five* examples. (More than 5-10 can sometimes confuse the model or waste context window space).
- **Examples:**
    - *Example 1:* Review: "The battery died in two hours." -> Sentiment: Negative
    - *Example 2:* Review: "Absolutely stunning display and fast processor!" -> Sentiment: Positive
    - *Example 3:* Review: "It works fine, but it's a bit heavy to carry around." -> Sentiment:
    - *(The AI will output "Neutral" or "Mixed" because it learned the exact pattern and vocabulary you wanted from the previous shots).*
- **Why it matters:** It drastically improves accuracy for complex tasks like data extraction, nuanced classification, or highly specific formatting (like generating SQL queries or matching a highly specific brand voice) where explaining the rules via text is too difficult or ambiguous.

---

← [1. LLM Fundamentals](01-llm-fundamentals.md) · [Contents](README.md) · [3. LLM APIs and SDKs](03-llm-apis-and-sdks.md) →
