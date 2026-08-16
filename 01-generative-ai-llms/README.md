# Lesson 1: Generative AI and Large Language Models

## What I Learned

This lesson introduced the core concepts behind modern Generative AI and Large Language Models.

I learned how Generative AI differs from traditional machine learning, how LLMs generate text, and how concepts such as tokens, embeddings, Transformers, attention, and context windows fit together.

I also explored the difference between prompting, RAG, fine-tuning, tool calling, and agentic AI.

---

## Core Concepts

### Traditional Machine Learning vs Generative AI

Traditional machine learning is commonly used to make predictions from existing data.

Example:

```text
Customer Data
     ↓
Machine Learning Model
     ↓
Churn / No Churn
```

Generative AI instead produces new content based on patterns learned from data.

```text
Prompt
   ↓
Generative Model
   ↓
Generated Content
```

Generated content can include:

* Text
* Code
* Images
* Audio
* Video
* Structured data

---

### Large Language Models

A Large Language Model is a neural network trained on large amounts of tokenised data.

At a simplified level, an autoregressive LLM repeatedly predicts the next token.

```text
Input
  ↓
Predict Next Token
  ↓
Add Token to Context
  ↓
Predict Next Token
  ↓
Repeat
```

---

### Tokens

LLMs do not process text exactly as complete words and sentences.

Text is divided into smaller units called tokens.

For example:

```text
Artificial intelligence is powerful
```

may be represented conceptually as:

```text
["Artificial", " intelligence", " is", " powerful"]
```

Tokens are converted into numerical representations before being processed by the model.

---

### Embeddings

Embeddings are numerical vector representations of information.
# Lesson 1: Generative AI and Large Language Models

## What I Learned

This lesson introduced the core concepts behind modern Generative AI and Large Language Models.

I learned how Generative AI differs from traditional machine learning, how LLMs generate text, and how concepts such as tokens, embeddings, Transformers, attention, and context windows fit together.

I also explored the difference between prompting, RAG, fine-tuning, tool calling, and agentic AI.

---

## Core Concepts

### Traditional Machine Learning vs Generative AI

Traditional machine learning is commonly used to make predictions from existing data.

Example:

```text
Customer Data
     ↓
Machine Learning Model
     ↓
Churn / No Churn
```

Generative AI instead produces new content based on patterns learned from data.

```text
Prompt
   ↓
Generative Model
   ↓
Generated Content
```

Generated content can include:

* Text
* Code
* Images
* Audio
* Video
* Structured data

---

### Large Language Models

A Large Language Model is a neural network trained on large amounts of tokenised data.

At a simplified level, an autoregressive LLM repeatedly predicts the next token.

```text
Input
  ↓
Predict Next Token
  ↓
Add Token to Context
  ↓
Predict Next Token
  ↓
Repeat
```

---

### Tokens

LLMs do not process text exactly as complete words and sentences.

Text is divided into smaller units called tokens.

For example:

```text
Artificial intelligence is powerful
```

may be represented conceptually as:

```text
["Artificial", " intelligence", " is", " powerful"]
```

Tokens are converted into numerical representations before being processed by the model.

---

### Embeddings

Embeddings are numerical vector representations of information.

Conceptually:

```text
"machine learning"

        ↓

[0.18, -0.42, 0.91, ...]
```

Semantically related concepts tend to have similar vector representations.

Embeddings are important for:

* Semantic search
* Recommendation systems
* Document retrieval
* Retrieval-Augmented Generation
* Similarity search

---

### Transformers

Most modern LLMs use the Transformer architecture.

A simplified pipeline is:

```text
Text
 ↓
Tokens
 ↓
Embeddings
 ↓
Transformer Layers
 ↓
Attention
 ↓
Next-Token Probabilities
 ↓
Generated Output
```

---

### Attention

Attention allows the model to determine which parts of the input are important when interpreting or generating a particular token.

This helps the model understand relationships between words and maintain context across a sequence.

---

### Context Window

The context window is the amount of information that a model can consider during an interaction.

It may include:

```text
System Instructions
        +
Conversation History
        +
Documents
        +
Tool Results
        +
Current Prompt
```

Because context windows are limited, modern AI applications often use techniques such as RAG, memory, databases, and external tools.

---

### Training vs Inference

**Training** is the process through which a model learns and updates its parameters.

```text
Training Data
     ↓
Prediction
     ↓
Calculate Error
     ↓
Backpropagation
     ↓
Update Parameters
```

**Inference** occurs when an already-trained model is used to generate a result.

```text
Prompt
   ↓
Trained Model
   ↓
Response
```

Using an existing LLM normally involves inference rather than training the model again.

---

## Important AI Engineering Distinctions

### Prompt Engineering

Changes the instructions or context given to the model.

### Retrieval-Augmented Generation

Retrieves relevant external information and provides it to the model during inference.

```text
Question
   ↓
Retrieve Relevant Information
   ↓
LLM
   ↓
Grounded Response
```

### Fine-Tuning

Performs additional training to influence how a model behaves on particular tasks.

### Tool Calling

Allows a model to interact with external functionality such as:

* APIs
* Databases
* Search engines
* Python functions
* Email systems
* Calendars

### Agentic AI

Combines an LLM with tools, instructions, state, and control logic to complete multi-step goals.

```text
Goal
 ↓
Reason
 ↓
Plan
 ↓
Use Tool
 ↓
Observe Result
 ↓
Decide Next Step
 ↓
Repeat
 ↓
Final Result
```

---

## Generative AI vs Genetic Algorithms

Generative AI and Genetic Algorithms are different concepts.

Generative AI focuses on generating new content.

Genetic Algorithms are optimisation algorithms inspired by biological evolution.

A typical Genetic Algorithm follows:

```text
Initial Population
        ↓
Evaluate Fitness
        ↓
Selection
        ↓
Crossover
        ↓
Mutation
        ↓
New Population
        ↓
Repeat
```

Genetic Algorithms will be covered separately later in this learning journey.

---

## Practical Exercise

### `mini_language_model.py`

For the practical component of this lesson, I built a small statistical language model.

The program:

1. Tokenises a small text corpus.
2. Learns which words commonly follow other words.
3. Counts word transitions.
4. Calculates simple probabilities.
5. Samples a possible next word.
6. Repeats the process to generate text.

The program demonstrates the basic idea behind autoregressive generation:

```text
Previous Input
      ↓
Probability Distribution
      ↓
Choose Next Token
      ↓
Add Token to Input
      ↓
Repeat
```

This is **not an LLM**.

Modern LLMs use neural networks, Transformers, attention mechanisms, embeddings, large vocabularies, and enormous training datasets.

The exercise was designed to demonstrate the underlying next-token generation concept in a simple and understandable way.

---

## Key Takeaway

The main LLM pipeline introduced in this lesson is:

```text
Text
 ↓
Tokens
 ↓
Embeddings
 ↓
Transformer
 ↓
Attention
 ↓
Next-Token Probabilities
 ↓
Generated Output
```

The broader progression toward modern AI systems is:

```text
Generative AI
      ↓
Large Language Models
      ↓
LLM Applications
      ↓
RAG + Tools + Memory
      ↓
AI Agents
      ↓
Multi-Agent Systems
```

Understanding this progression provides the foundation for the rest of this repository.

---

## Files

* `README.md` — Summary of Lesson 1 concepts.
* `mini_language_model.py` — Simple autoregressive language-model demonstration.

---

## Status

**Lesson 1 completed.**

