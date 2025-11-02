# LangGraph & LangChain Project Series  

## 📖 Introduction
As employees in the 21st century, we face an inevitable reality: artificial intelligence is transforming the workplace faster than ever before. I don’t believe it’s a matter of if our jobs will be affected — it’s a matter of when. In China, AI-powered robots now assemble cars in “dark factories,” where production continues around the clock without the need for human supervision — or even lights. Intelligent AI agents aren’t just coming; they’re already here.

As a cybersecurity analyst, I recognize that my field is no exception. Many of the tasks we perform today can and likely will be automated by intelligent agents. That’s why I believe it’s essential for professionals like us to understand how these systems work — not to fear them, but to learn how to design, maintain, and secure them. If we want to stay relevant in the era of autonomous agents, we need to be the ones building and managing them.

This project marks my first step in that direction. Through a series of experiments and workflows, I aim to build hands-on familiarity with LangGraph and LangChain, exploring how agentic systems can be designed, orchestrated, and applied to real-world problems.

A special thanks to [Mr. Hacker Loi](https://youtu.be/rfPsjqNbWyA?si=iMXEsQFLkX4wr_v4).

## 🔍 Overview
According to [LangChain's official documentation](https://docs.langchain.com/oss/python/langgraph/workflows-agents):

> **Workflows** have predetermined code paths and are designed to operate in a certain order.
> **Agents** are dynamic and define their own processes and tool usage.

This repository is part of my ongoing series of projects designed to build **familiarity with LangGraph** and **LangChain**—two powerful frameworks for developing **AI-driven systems** that combine reasoning, memory, and structured task orchestration.  

Each part of this series focuses on a specific aspect of intelligent agent design, starting from static, rule-based workflows and progressing toward dynamic, autonomous AI agents.

## 🎯 Goals

- Understand how **LangGraph** structures logic using **states**, **nodes**, and **edges**  
- Learn to design **workflows** that control and monitor AI behavior predictably  
- Gain practical experience integrating **LLMs**, **retrieval tools**, and **self-evaluation loops**  
- Transition from **workflows** → **multi-agent systems** capable of decision-making and collaboration  

## 📂 Project Structure

| Part | Title | Description | Link |
|------|--------|--------------|------|
| 1 | **Workflows** | Implements a LangGraph-based **IT support ticket system** using a structured workflow. The system classifies support tickets, retrieves knowledge base entries, drafts responses, and self-evaluates them. | [Part 1: Workflows]([Workflows/readme.md](https://github.com/dsuyu1/workflows-and-agents-with-langgraph/tree/119a820aa378a5af64b4b607d75dd58e2277a3ad/Part%201%3A%20Workflows)) |
| 2 | **Agents** *(coming soon)* | Expands on the previous system by introducing **autonomous agents** capable of dynamic reasoning, task delegation, and self-correction. | _In progress_ |
