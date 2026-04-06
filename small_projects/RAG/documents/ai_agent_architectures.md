# AI Agent Architectures — How Agents Think and Act

## What is an AI Agent?

An AI agent is a system that perceives its environment, makes decisions, and takes actions to achieve a goal. Unlike a simple chatbot that only responds to questions, an agent can plan multi-step tasks, use tools, remember context, and adapt its behavior based on feedback.

---

## Core Components of an AI Agent

### 1. Brain (LLM)
The large language model is the reasoning core of the agent. It interprets inputs, decides what to do next, and generates outputs. Popular choices include GPT-4o, Claude, and Gemini.

### 2. Memory
Memory allows the agent to retain information across steps or sessions. There are four types:

- **Sensory Memory**: The immediate input — the current prompt or observation
- **Short-term Memory**: The conversation history within a session (context window)
- **Long-term Memory**: External storage like vector databases that persist across sessions
- **Episodic Memory**: Specific past experiences the agent can recall and reference

### 3. Tools
Tools extend the agent's capabilities beyond text generation. Common tools include:

- Web search
- Code execution
- Calculator
- File reading and writing
- API calls (weather, calendar, email)
- Database queries

### 4. Planning
Planning is the agent's ability to break down a complex goal into smaller, manageable steps and execute them in order.

---

## Common Agent Architectures

### ReAct (Reasoning + Acting)
ReAct is one of the most widely used agent patterns. The agent alternates between reasoning about what to do and taking actions, observing the result of each action before deciding the next step.

**The ReAct loop:**
1. **Thought** — the agent reasons about the current situation
2. **Action** — the agent calls a tool
3. **Observation** — the agent reads the tool's output
4. **Repeat** until the goal is achieved

ReAct is powerful because it makes the agent's reasoning transparent and allows it to course-correct based on real feedback from tools.

### Plan and Execute
Instead of reasoning one step at a time, the agent first creates a full plan for the entire task, then executes each step sequentially. This is useful for complex, multi-step tasks where the overall strategy is important to get right upfront.

**Steps:**
1. Planner agent creates a step-by-step plan
2. Executor agent carries out each step
3. Results are aggregated at the end

### Reflection
A reflection agent reviews its own output before returning it. After generating a response or completing a task, it critiques its own work and iterates to improve quality. This is especially useful for writing, coding, and research tasks.

### Multi-Agent Systems
Multiple specialized agents collaborate to solve a problem. Each agent has a specific role and they communicate by passing information between each other.

**Common roles:**
- **Orchestrator**: Coordinates the overall workflow
- **Researcher**: Gathers information
- **Critic**: Evaluates quality and flags issues
- **Writer**: Produces final output

---

## Agent Memory Strategies

### In-Context Memory
Storing everything in the LLM's context window. Simple but limited by the context window size and loses information between sessions.

### External Vector Memory
Storing information as embeddings in a vector database. The agent retrieves relevant memories using semantic search — this is the foundation of RAG systems.

### Summary Memory
Periodically summarizing older parts of a conversation to compress it while retaining the key information. Balances token efficiency with context preservation.

---

## Tool Use Patterns

### Sequential Tool Use
The agent uses tools one at a time in a fixed order. Simple and predictable, good for structured pipelines like the Morning Briefing Agent.

### Parallel Tool Use
The agent calls multiple tools simultaneously and waits for all results before proceeding. Faster for independent tasks like fetching weather and news at the same time.

### Conditional Tool Use
The agent decides which tool to call based on the current context. More flexible but requires better reasoning from the LLM.

---

## Popular Agent Frameworks

| Framework | Best For |
|---|---|
| LangChain | General purpose, rich ecosystem |
| LangGraph | Complex multi-agent workflows with state |
| AutoGen | Multi-agent conversations |
| CrewAI | Role-based multi-agent teams |
| Plain Python | Simple pipelines, full control |

---

## When to Use an Agent vs a Simple LLM Call

Use a **simple LLM call** when:
- The task is a single step
- No external data is needed
- The answer comes from the model's training knowledge

Use an **agent** when:
- The task requires multiple steps
- External tools or APIs are needed
- The model needs to react to feedback
- The task involves real-time or personalized data
