# Prompt Engineering — Techniques and Best Practices

## What is Prompt Engineering?

Prompt engineering is the practice of designing, refining, and optimizing the instructions given to a large language model (LLM) to reliably produce desired outputs. Since LLMs are highly sensitive to how they are prompted, small changes in wording can significantly affect the quality, format, and accuracy of responses.

---

## Why Prompt Engineering Matters

LLMs do not truly "understand" instructions the way humans do — they predict the most likely next token based on patterns in their training data. A well-crafted prompt steers the model toward the pattern that produces the most useful response for your use case.

---

## Core Prompting Techniques

### Zero-Shot Prompting
Asking the model to perform a task without providing any examples. The model relies entirely on its pre-trained knowledge.

**Example:**
```
Classify the sentiment of this review as Positive, Negative, or Neutral:
"The delivery was late but the product quality was excellent."
```

Zero-shot works well for common tasks but may struggle with nuanced or domain-specific ones.

---

### Few-Shot Prompting
Providing a few examples of the desired input-output behavior before asking the model to perform the task. This guides the model toward the expected format and reasoning style.

**Example:**
```
Classify sentiment:
Review: "Great product, fast shipping!" → Positive
Review: "Terrible experience, broken on arrival." → Negative
Review: "It's okay, nothing special." → Neutral

Review: "The battery life is amazing but the screen is dim." → ?
```

Few-shot prompting significantly improves performance on complex or specialized tasks.

---

### Chain-of-Thought Prompting (CoT)
Encouraging the model to reason step by step before giving a final answer. Adding "think step by step" or showing reasoning examples dramatically improves performance on math, logic, and multi-step problems.

**Example:**
```
Q: A store has 50 apples. They sell 30% on Monday and 20 more on Tuesday. How many are left?
A: Let's think step by step.
- 30% of 50 = 15 apples sold Monday
- Remaining after Monday: 50 - 15 = 35
- Sold Tuesday: 20
- Remaining: 35 - 20 = 15 apples
```

---

### System Prompts
A system prompt sets the overall persona, context, and constraints for the model at the beginning of a conversation. It is the most powerful way to shape model behavior consistently across an entire session.

**Best practices for system prompts:**
- Define the model's role clearly ("You are a helpful Python tutor")
- Set constraints ("Answer only using the provided context")
- Specify output format ("Always respond in bullet points")
- Set the tone ("Be concise and friendly")

---

### Role Prompting
Assigning a specific persona or role to the model to leverage domain-specific knowledge and tone.

**Example:**
```
You are an experienced cardiologist. Explain the difference between 
systolic and diastolic blood pressure in simple terms for a patient.
```

Role prompting is effective for technical domains, creative writing, and adapting communication style.

---

### Instruction Following with Constraints
Being explicit about what the model should and should not do.

**Example:**
```
Summarize the following article in exactly 3 bullet points.
Do not include any information not present in the article.
Do not use the word "however".
```

Explicit constraints reduce unwanted model behaviors and improve output consistency.

---

### Output Format Specification
Telling the model exactly how to format its output — JSON, markdown, bullet points, tables, etc.

**Example:**
```
Extract the key information from the following job description and 
return it as a JSON object with these fields:
- job_title
- required_skills (list)
- experience_years (number)
- location
```

This is especially important when building applications that need to parse model output programmatically.

---

## Advanced Techniques

### Self-Consistency
Generate multiple responses to the same prompt and pick the most common answer. Useful for reasoning tasks where the model might arrive at different conclusions on different runs.

### Tree of Thoughts
A more advanced form of chain-of-thought where the model explores multiple reasoning paths simultaneously, evaluates each branch, and selects the most promising one. Useful for complex planning and problem-solving tasks.

### RAG Prompting
Injecting retrieved documents into the prompt as context, instructing the model to answer based only on that context. Reduces hallucinations and grounds responses in factual, up-to-date information.

```
Answer the question using only the context below. 
If the answer is not in the context, say "I don't know."

Context:
{retrieved_documents}

Question: {user_question}
```

---

## Common Mistakes to Avoid

| Mistake | Better Approach |
|---|---|
| Vague instructions | Be specific about task, format, and constraints |
| Too long a prompt | Keep it concise — every token costs attention |
| No examples for complex tasks | Add 2-3 few-shot examples |
| Assuming the model knows your context | Provide all necessary background information |
| Not iterating | Treat prompting as an experiment — test and refine |

---

## Prompt Engineering Workflow

1. **Start simple** — try zero-shot first
2. **Evaluate output** — is it accurate, formatted correctly, and consistent?
3. **Add examples** — if zero-shot fails, add few-shot examples
4. **Add reasoning** — if accuracy is low, add chain-of-thought
5. **Tighten constraints** — if format is inconsistent, add explicit formatting rules
6. **Test edge cases** — try unusual inputs to find failure modes
7. **Document what works** — save effective prompts for reuse
