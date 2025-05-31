# llm-design-patterns

- https://www.anthropic.com/engineering/building-effective-agents
- https://www.philschmid.de/agentic-pattern

| Contents                 |
|--------------------------|
| [Evaluator Optimiser (Reflection)](#evaluator-optimiser-reflection) |
| [Full Agent](#full-agent) |
| [Memory](#memory) | 
| [Multimodal Input/Output](#multimodal-inputoutput) |
| [Orchestrator and Workers](#orchestrator-and-workers) |
| [Parallel Processing](#parallel-processing) |
| [Prompt Chaining](#prompt-chaining) |
| [Retrieval-Augmented Generation (RAG)](#retrieval-augmented-generation-rag) |
| [Routing](#routing) |
| [Tools (function-calling)](#tools-function-calling) |


## Evaluator Optimiser (Reflection)

## Full Agent

## Memory

## MultiModal Input/Output

```bash
# example of including an image in the prompt:
uv run python -m multimodal.image_input
```

## Orchestrator and Workers

## Parallel Processing

There are a few different ways to run code in parallel/concurrently. Examples are multi-core, multi-thread, async and greenlets (not an exhaustive list). There are different tradeoffs associated with each. Here is an example using the async OpenAI client (over 100 chat completions, I measured this approach to be 32x faster than synchronous API calls in a simple for loop):

```bash
uv run python parallel_processing/async.py
```

## Prompt Chaining

*Prompt Chaining* refers to sequential LLM calls, where the output of one feeds into the prompt of the next.

![source: https://www.anthropic.com/engineering/building-effective-agents](./static/anthropic_prompt_chaining.png)

Here is a simple 2-step example:
```bash
uv run python -m prompt_chaining.step_back_prompting_example
```

## Retrieval-Augmented Generation (RAG)

## Routing

## Tools (function-calling)

