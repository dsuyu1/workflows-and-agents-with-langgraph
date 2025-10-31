# Workflows and AI Agents with LangGraph
In this project, I build an agentic workflow with Ollama and LangGraph. I will use [this](https://youtu.be/mRx12jkugTE?si=48OSoke3ebptm8gn) tutorial from Venelin Valkov to get started. After the system is complete, I will add my own additions to make the system unique. 

Feel free to skip forward to the end to see my final product!

## 1. Introduction
I'll build two implementations of the IT support ticket system - one with workflows and the otehr with AI agents to understand the differences and use cases.

The LangGraph library is focused around building workflows and AI agents.

As previously mentioned, our use case is an **intelligent support ticket triage**.

### Problem Statement
Our company is drowning in support tickets. The IT team could use some help! ⚠️


### Proposed Solution
Intelligent ticket processing that:
- Classifies the ticket type (technical, billing, general)
- Retrieves relevant solutions from a knowledge base
- Drafts helpful responses using found information
- Reviews and revises its own work until quality standards are met

The system is rule-based enough to be predictable, complex enough to need orchestration.

### Why don't we just use simple LLM calls?
Simple LLM calls that use if/else statements and loops are fine for systems with 1-2 nodes, but managing and maintaining a complex system with just if/else and loops is too cumbersome. Business logic gets buried, and manual state management is a pain.

LangGraph gives us **three** building blocks to design AI systems like production workflows:
- **State**: This is the system's memory 🧠. The state tracks everything from `ticket_text` to `draft_response`.
- **Nodes**: Nodes are Python functions that do one job well. For example `classify_ticket`, `draft_response`, `evaluate_draft`.
- **Edges**: The paths between the nodes. They are direct connections or conditional logic.

### Workflows vs. Agents: Who's in Control?
| Workflows | Agents |
| --- | --- |
| Developer controls every step | LLM decides the strategy |
| Fixed, predictable path | Dynamic ReAct loop (Reason -> Act -> Observe) |
| Perfect for compliance, data pipelines | Perfect for research, creative problem-solving |
| Rock solid reliability | Adaptive intelligence |

Within a workflow, as a developer, we can control every step of the process. It is easy to debug too!

AI agents are all the rage right now, but even with powerful LLMs, if we have a specific workfllow that needs to be done in a specific order, agents are not able to follow through how we might want them to.

### Human-in-the-Loop: Our Safety Net
Autonomous agents are powerful but can be expensive when they go wrong. We need human oversight, especially in enterprise/production environments. 

We can **add strategic checkpoints:**
- Before **high impact** actions (escalations, billing)
- When AI is **uncertain** (low confidence scores)
- Final quality gate (customer-facing responses)

For example, we can have a function that freezes the graph and waits for human review right before it escalates a ticket. 

Human-in-the-loop takes a cool AI demo app to something you can actually use in production. With humans "in the loop," we can relax knowing that AI won't go crazy. :relieved:

# 2. Implementation
We start by importing everything we need. Most of them come from LangChain. The LangGraph imports will help us build our state graphs.
```python
# Used to easily create classes for storing data.
from dataclasses import dataclass, field
# Provides type hints for better code readability and maintainability.
from typing import Annotated, List, TypedDict

# Used to display images and other rich output in IPython/Colab.
from IPython.display import Image, display
# Initializes a chat model for language model interactions.
from langchain.chat_models import init_chat_model
# Provides fast and efficient embeddings for text.
from langchain_community.embeddings import FastEmbedEmbeddings
# Represents a document object, often used in retrieval systems.
from langchain_core.documents import Document
# Represents different types of messages in a conversation.
from langchain_core.messages import AnyMessage, HumanMessage
# Creates templates for generating chat prompts.
from langchain_core.prompts import ChatPromptTemplate
# Decorator to define a function as a tool for language models.
from langchain_core.tools import tool
# An in-memory vector store for storing and searching vector embeddings.
from langchain_core.vectorstores import InMemoryVectorStore
# Components for building state graphs in LangGraph.
from langgraph.graph import END, StateGraph
# Function to add messages to the state in LangGraph.
from langgraph.graph.message import add_messages
# A pre-built node in LangGraph for executing tools.
from langgraph.prebuilt import ToolNode
```
I'll be using `gpt-oss`'s latest model provided by Ollama. According to Ollama, `gpt-oss` is
> "OpenAI’s open-weight models designed for powerful **reasoning**,** agentic tasks**, and versatile developer use cases."

Seem's like this model fits our use case pretty well. Using Ollama is great because I have it running locally on my machine.

## Workflow
Our intelligent system workflow starts by defining a **state**. We'll use a `dataclass` to define our state. In LangChain, a `dataclass` is utilized as a method for **defining structure of data**, particularly within the context of managing state in LangGraph. 

```python
@dataclass # defines structure of a our LangChain states
class TicketTriageState:
  ticket_text: str
  classification: str= "" #
  retrieved_docs: List[Document] = field(default_factory=lambda: []) # subject to change
  draft_response: str = "" 
  evaluation_feedback: str = "" 
  revision_count: int = 0 # sets revision_count to 0 to start
```

## Nodes
<img src="nodes_graphic.png" alt="nodes" width="500"/>

In LangGraph, a **node** is a _function_ that represents a single unit of computation or a specific step in a workflow.


