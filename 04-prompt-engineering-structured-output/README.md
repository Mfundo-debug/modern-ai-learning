# Lesson 4: Prompt Engineering and Structured Outputs

## What I Learned

This lesson focused on how to control the behaviour and output of a Large Language Model more reliably.

I learned how prompt engineering differs from structured output design, how developer instructions should remain reusable, and how schemas can enforce output structure more effectively than relying only on natural-language instructions.

I also learned that reliable AI applications usually combine three layers of control:

```text
Prompt
  ↓
Schema
  ↓
Application Validation
```

---

## Core Concepts

### Prompt Engineering

Prompt engineering is the process of designing instructions that clearly communicate what a model should do.

A useful prompt can include:

```text
ROLE
+
AUDIENCE
+
GOAL
+
CONSTRAINTS
+
QUALITY REQUIREMENTS
+
OUTPUT EXPECTATIONS
```

The objective is not to make prompts unnecessarily long.

The objective is to reduce ambiguity and make successful completion of the task clear.

---

## Weak vs Better Prompting

A weak prompt may be:

```text
Explain RAG.
```

This leaves many decisions to the model.

For example:

```text
How technical should the answer be?
Who is the learner?
How long should the explanation be?
Should an example be included?
```

A stronger prompt specifies the intended behaviour.

For example:

```text
Explain Retrieval-Augmented Generation to a learner
who understands basic machine learning.

Include:
- a concise definition
- the role of embeddings
- the retrieval process
- one practical example

Keep the explanation technically accurate and concise.
```

---

## Developer Instructions vs User Input

A reusable AI application should separate application behaviour from the user's current task.

### Developer Instructions

These define how the application should behave.

Example:

```text
You are a technical AI tutor.

Explain concepts accurately and clearly.

Assume the learner understands Python and basic
machine learning.
```

### User Input

This defines the current task.

Example:

```text
Explain vector databases.
```

The structure is therefore:

```text
Developer Instructions
        ↓
Define application behaviour

User Input
        ↓
Define the current task
```

A specific topic such as `for loops in Python` should normally be supplied by the user rather than hard-coded into reusable developer instructions.

---

## Zero-Shot Prompting

Zero-shot prompting provides instructions without giving examples.

Example:

```text
Classify the following concept as beginner,
intermediate, or advanced:

Transformer self-attention
```

---

## Few-Shot Prompting

Few-shot prompting includes examples that demonstrate the expected pattern.

Example:

```text
Classify programming concepts by difficulty.

Examples:

Variables → beginner
Functions → beginner
Recursion → intermediate
Metaclasses → advanced

Concept:
Decorators
```

Examples can help when a task contains specialised patterns or ambiguous requirements.

However, examples should be added because they improve performance, not simply because more prompt content appears more sophisticated.

---

## Structured Outputs

Free-form text is useful for conversation but can be difficult for software to process reliably.

Instead of receiving:

```text
Self-attention is an important mechanism used by...
```

an application may require:

```json
{
  "topic": "Self-attention",
  "definition": "A mechanism...",
  "difficulty": "intermediate",
  "keywords": [
    "query",
    "key",
    "value"
  ]
}
```

Structured output allows application code to work directly with individual fields.

---

## JSON vs Schema-Constrained Output

Valid JSON does not necessarily mean the structure is correct for an application.

For example:

```json
{
  "name": "Self-attention",
  "level": "Medium"
}
```

is valid JSON.

However, an application may expect:

```json
{
  "topic": "...",
  "definition": "...",
  "difficulty": "...",
  "keywords": []
}
```

The distinction is:

```text
JSON
 ↓
Valid data format

Schema
 ↓
Rules describing the required structure
```

---

## Pydantic

Pydantic allows Python applications to define structured data models.

Example:

```python
from pydantic import BaseModel


class TopicExplanation(BaseModel):
    topic: str
    definition: str
    difficulty: str
    keywords: list[str]
```

This defines the expected fields and their data types.

---

## Using Literal Constraints

A normal string field such as:

```python
difficulty: str
```

could theoretically contain any string.

For example:

```text
beginner
intermediate
advanced
unknown
potato
```

If only specific values are valid, a `Literal` type can define the allowed values.

```python
from typing import Literal


difficulty: Literal[
    "beginner",
    "intermediate",
    "advanced"
]
```

This creates a stronger application contract.

---

## Practical Application

### `structured_llm_app.py`

The application created in this lesson defines a structured response model using Pydantic.

Example:

```python
from typing import Literal
from pydantic import BaseModel


class TopicExplanation(BaseModel):
    topic: str

    definition: str

    difficulty: Literal[
        "beginner",
        "intermediate",
        "advanced"
    ]

    keywords: list[str]
```

The model request then uses the schema to produce a parsed response.

Conceptually:

```text
User Input
    ↓
Developer Instructions
    ↓
LLM
    ↓
Schema
    ↓
Parsed Python Object
    ↓
Application Logic
```

---

## Prompt vs Schema

One of the most important lessons from this exercise is that not every rule should exist only inside the prompt.

For example:

```text
"Difficulty must be beginner."
```

can be communicated in a prompt.

However, if the application truly requires that value, it is stronger to encode it programmatically.

Example:

```python
difficulty: Literal["beginner"]
```

The principle is:

```text
Prompt
 ↓
Describe desired behaviour

Schema
 ↓
Enforce structure and allowed values
```

---

## Application-Level Validation

A schema cannot enforce every type of requirement.

For example:

```text
Definition must contain no more than 80 words.
```

or:

```text
The generated code must be valid Python.
```

may require additional application logic.

This introduces a third layer:

```text
Prompt
 ↓
Tell the model what good output should look like

Schema
 ↓
Enforce structure and allowed values

Application Validation
 ↓
Verify additional rules
```

Examples of application validation include:

* Counting words
* Compiling generated Python code
* Checking numeric ranges
* Verifying required business rules
* Checking retrieved information
* Rejecting incomplete results

---

## Prompt Engineering Challenge

The practical challenge involved designing instructions for a first-year programming tutor.

The output needed to include:

```text
topic
difficulty
definition
example
common_mistake
keywords
```

A key correction from the exercise was that the specific topic should not be embedded in reusable developer instructions.

Instead:

```text
Developer Prompt
 ↓
Defines how all programming topics should be explained

User Input
 ↓
Provides the specific programming concept
```

For example:

```text
Developer:
You are a programming tutor for first-year students.

User:
Explain for loops in Python.
```

---

## Example Improved Developer Instructions

```text
You are a programming tutor for first-year university students.

Explain the programming concept provided by the user at a
beginner level.

The learner has limited programming experience, so use clear,
accessible language while remaining technically accurate.

Provide:
- a concise definition;
- one valid and practical Python example;
- one realistic beginner mistake;
- between 3 and 5 important keywords.

Ensure that any Python code is syntactically valid and directly
relevant to the concept being explained.
```

---

## Structured AI Engineering

This lesson introduced an important progression:

```text
Natural-Language Prompt
        ↓
Model Generation
        ↓
Structured Schema
        ↓
Parsed Object
        ↓
Validation
        ↓
Application Logic
```

This is significantly more reliable than treating an LLM as a simple chatbot.

---

## Connection to Previous Lessons

### Lesson 1

Learned how autoregressive language models generate tokens.

### Lesson 2

Learned how self-attention creates context-aware representations.

### Lesson 3

Used a real LLM through an API.

### Lesson 4

Learned how to control model behaviour and output structure.

The progression is now:

```text
Understand the Model
        ↓
Access the Model
        ↓
Control the Model
        ↓
Use the Model as a Software Component
```

---

## Key Takeaway

A reliable AI application should not depend entirely on the model following natural-language instructions perfectly.

A stronger design combines:

```text
PROMPT
"What should the model do?"

        ↓

SCHEMA
"What structure and values are allowed?"

        ↓

VALIDATION
"Does the result actually satisfy the application rules?"

        ↓

APPLICATION LOGIC
"What should the software do with the result?"
```

This distinction becomes increasingly important when building RAG systems, tool-using applications, and AI agents.

---

## Files

* `README.md` — Summary of prompt engineering and structured output concepts.
* `structured_llm_app.py` — Structured LLM application using Pydantic.

---

## Status

**Lesson 4 completed.**
