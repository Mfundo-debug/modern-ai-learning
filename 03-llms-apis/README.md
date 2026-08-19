# Lesson 3: Using a Large Language Model Through an API

## What I Learned

This lesson moved from understanding how LLMs work internally to using a real Large Language Model programmatically.

I learned how a Python application communicates with an external LLM through an API, how API authentication works, why API keys must be protected, and how to send dynamic user input to a model.

I also learned how to diagnose a common configuration problem involving environment variables and how API-based applications should handle failures more gracefully.

---

## Core Concepts

### What is an API?

An API allows one software application to communicate with another service.

In an LLM application, the architecture can be represented as:

```text
User
 ↓
Python Application
 ↓
LLM API
 ↓
Model
 ↓
API Response
 ↓
Python Application
 ↓
User
```

The LLM does not run locally inside the Python script.

The Python program sends a request to a remote service, where the model performs inference and returns a response.

---

## API Authentication

An API key identifies and authenticates the application making the request.

Conceptually:

```text
Python Application
      +
API Key
      ↓
LLM API
```

API keys should never be hard-coded directly into source code that may be committed to GitHub.

Avoid:

```python
api_key = "my-secret-key"
```

A safer approach is to store the key as an environment variable.

For example:

```text
OPENAI_API_KEY
```

The application can then access the credential without exposing it inside the source code.

---

## Environment Variables

Environment variables allow configuration information to be stored outside application code.

The application can therefore remain:

```text
Code
 +
External Configuration
```

rather than:

```text
Code
 +
Embedded Secrets
```

This is important for security and deployment.

A practical lesson from this exercise was that environment-variable commands differ between Windows Command Prompt and PowerShell.

For example:

```text
Command Prompt:
echo %OPENAI_API_KEY%
```

and:

```text
PowerShell:
echo $env:OPENAI_API_KEY
```

are different command syntaxes.

---

## Creating the API Client

The OpenAI Python library provides a client object that is used to communicate with the API.

```python
from openai import OpenAI

client = OpenAI()
```

When configured correctly, the client can retrieve the API key from the environment.

---

## Sending a Model Request

A simple model request can be created using:

```python
response = client.responses.create(
    model="gpt-5.6",
    input="Explain machine learning in one sentence."
)
```

The request contains two important pieces of information:

```text
model
  ↓
Which model should perform the task?

input
  ↓
What should the model respond to?
```

---

## Retrieving the Response

The generated text can be accessed through:

```python
print(response.output_text)
```

The simplified request-response flow is:

```text
Python Input
     ↓
API Request
     ↓
LLM Inference
     ↓
API Response
     ↓
response.output_text
```

---

## Instructions vs User Input

LLM applications often separate permanent application behaviour from dynamic user input.

For example:

```python
response = client.responses.create(
    model="gpt-5.6",

    instructions="""
    You are an AI tutor.
    Explain technical concepts clearly.
    Assume the learner understands Python and basic machine learning.
    """,

    input=question
)
```

The distinction is:

```text
Instructions
     ↓
How should the model behave?

Input
     ↓
What task should the model perform?
```

This is important because application developers normally define the behaviour of the AI system, while users provide the actual questions or tasks.

---

## Interactive LLM Application

I extended the program so that the user can enter a question dynamically.

```python
question = input("Ask the AI a question: ")
```

The value entered by the user is then passed to the model.

The application architecture becomes:

```text
User
 ↓
input()
 ↓
Python Application
 ↓
LLM API
 ↓
Model
 ↓
Generated Response
 ↓
User
```

This turns the script from a fixed API test into a small interactive LLM application.

---

## Practical Implementation

### `first_llm_app.py`

The completed application follows this general structure:

```python
from openai import OpenAI

client = OpenAI()

question = input("Ask the AI a question: ")

try:
    response = client.responses.create(
        model="gpt-5.6",

        instructions="""
        You are an AI tutor.
        Explain technical concepts clearly.
        Assume the learner understands Python and basic machine learning.
        Give concise but technically accurate answers.
        """,

        input=question
    )

    print("\nAI RESPONSE:\n")
    print(response.output_text)
except Exception as error:
    print("\nThe request could not be completed.")
```

---

## Error Handling


Examples include:

* Missing credentials
* Invalid API keys
* Rate limits
* Network problems
* Invalid model names
* Incorrect request parameters
* Service interruptions
Instead of allowing the application to crash immediately, errors can be handled using:

```python
try:
    # API request

except Exception as error:
    # Handle error
```
This introduces an important AI engineering principle:

```text
A model call is an external dependency.
```

Applications should therefore expect that requests may fail.


## Troubleshooting Experience

During this lesson, the application initially returned a missing-credentials error.

The issue was not caused by the model or Python code.


This demonstrated that AI application failures may occur at different layers:

```text
Application Logic
       ↓
Dependencies
       ↓
Configuration
       ↓
Authentication
API Service
       ↓
Model
```

Effective debugging requires identifying which layer is actually failing.

---

## Connection to Previous Lessons

### Lesson 1

I learned that autoregressive language models generate output by repeatedly predicting the next token.

### Lesson 2
I explored how Transformer self-attention produces context-aware token representations.

### Lesson 3

I used a real LLM through an API.

The combined mental model is now:

```text
Python Application
       ↓
API Request
       ↓
LLM
       ↓
Tokenisation
       ↓
Embeddings
       ↓
Transformer Layers
       ↓
Self-Attention
       ↓
Next-Token Generation
       ↓
API Response
       ↓
Python Application
```

---

## Key Takeaway

An LLM application is more than a model.

Even a basic application involves several components:

```text
User
 ↓
Application Code
 ↓
Configuration
 ↓
Authentication
 ↓
API
 ↓
Model
 ↓
Response Handling
```

Building reliable AI systems therefore requires both AI knowledge and software engineering skills.

---

## Files

* `README.md` — Summary of LLM API concepts.
* `first_llm_app.py` — Interactive Python application using a real LLM API.

---

## Status

**Lesson 3 completed.**

       ↓
The problem was that the environment variable had been created but was not visible to the current terminal session.
---


* Insufficient API credits
External AI services can fail for many reasons.
    print(error)

