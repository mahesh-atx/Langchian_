# 1. LLM Fundamentals

[Home](README.md) · Part 1 of 14

> How the model itself works: tokens, context windows, temperature, sampling, and the failure modes to expect.

**In this part:** Tokens · Context windows · Temperature · Top-p / nucleus sampling · Hallucinations · Limitations

---

## 1. Tokens (The Building Blocks)
**What they are:** A token is the fundamental unit of text that an LLM reads and generates. It is not always a whole word; it is often a fragment of a word, a single character, or a symbol.
- **How it works:** LLMs do not see text the way humans do. They use a "tokenizer" to chop text into numbers (IDs) that the AI can process.
- **Examples:**
    - The word "apple" is one token.
    - The word "applepie" might be two tokens: "apple" and "pie".
    - Common suffixes like "-ing" or "-tion" are usually single tokens.
- **Rule of Thumb:** For English text, **1 token is roughly 0.75 words** (or about 4 characters).
- **Why it matters:** You pay for LLM usage based on tokens, and your context limit is measured in tokens.

## 2. Context Windows (Short-Term Memory)
**What it is:** The context window is the maximum amount of text the model can consider at one time. This includes both your prompt (input) and the model's response (output).
- **How it works:** Think of it like a physical window sliding over a book. The AI can only "see" what is currently inside the frame.
- **Size:** Older models had windows of 2,000 to 4,000 tokens. Modern models (like GPT-4o or Claude 3.5 Sonnet) can handle 128,000 to 1,000,000+ tokens (the equivalent of several full-length novels or large codebases).
- **Why it matters:** If you ask a question that requires data from outside the "window," the model simply cannot see it.
- **The "Lost in the Middle" Problem:** Research shows that while LLMs are good at remembering the very beginning and the very end of a long context window, they often "forget" or ignore information placed in the middle.

## 3. Temperature (The Creativity Dial)
**What it is:** A setting (usually from 0 to 2) that controls the randomness of the model's output. It essentially tells the model how "risky" it should be when choosing the next word.
- **How it works:** LLMs generate text by assigning a probability to every possible next word.
    - **Low Temperature (0.0 - 0.3):** The model almost always picks the most likely word. The output is highly deterministic, factual, and repetitive. Best for coding, math, and data extraction.
    - **Medium Temperature (0.5 - 0.7):** A balance of coherence and variety. Good for general conversation.
    - **High Temperature (0.8 - 1.2+):** The model is forced to consider less likely words. The output becomes highly creative, unpredictable, and sometimes nonsensical. Best for brainstorming or creative writing.

## 4. Top-p / Nucleus Sampling (The Vocabulary Filter)
**What it is:** Another way to control randomness, often used alongside Temperature. It limits the pool of words the model is allowed to choose from.
- **How it works:** "p" is a probability threshold (from 0 to 1).
    - If you set **Top-p = 0.9**, the model looks at the most likely next words and stops adding them to its "menu" once their combined probabilities reach 90%.
    - It ignores the remaining 10% of highly unlikely words.
    - If you set **Top-p = 0.1**, the model is only allowed to pick from the top 10% most likely next words, making it very focused.
- **Temperature vs. Top-p:** Temperature changes the *shape* of the probabilities (making unlikely words more likely). Top-p draws a hard *cutoff line* (banning the bottom-tier words entirely).

## 5. Hallucinations (Confident Lies)
**What they are:** When an LLM generates plausible-sounding but entirely incorrect, fabricated, or nonsensical information.
- **Why it happens:** LLMs are not databases of facts; they are "stochastic parrots." They are designed to predict the next statistically probable word in a sentence. If a sentence *sounds* like it should end with a specific legal citation or historical date, the model will invent one, even if it doesn't exist.
- **Examples:**
    - Inventing fake court cases or academic papers with real-looking authors.
    - Creating non-existent software libraries and writing code for them.
    - Providing confident answers to nonsensical questions (e.g., "What is the capital of the moon?").
- **How to fix it:** Use lower temperature settings, provide external context (Retrieval-Augmented Generation/RAG), or ask the model to "show its work" or admit if it doesn't know.

## 6. Limitations (The Broader Boundaries)
While LLMs are incredibly capable, they have fundamental flaws you must account for:
- **No True Reasoning:** They do not "think." They perform advanced pattern matching. They can solve a math problem they've seen before in their training data, but they may fail at a slightly altered logic puzzle.
- **Knowledge Cutoffs:** An LLM's knowledge is frozen at the time of its training. Unless it is connected to a live web search tool, it does not know about events that happened yesterday.
- **Bias:** Models inherit the biases, stereotypes, and toxicities present in the internet-scale data they were trained on.
- **Math & Logic Struggles:** While vastly improved, LLMs still struggle with complex multi-step arithmetic, spatial reasoning, and deep logical deduction compared to specialized calculators or human experts.
- **Lack of "Grounding":** They have no real-world experience. They don't know what an apple tastes like; they only know the statistical association of the word "apple" with words like "crunchy," "red," and "sweet."

---

[Contents](README.md) · [2. Prompt Engineering](02-prompt-engineering.md) →
