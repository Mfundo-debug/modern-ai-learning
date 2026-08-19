# Lesson 2: Transformers and Attention

## What I Learned

This lesson explored how Transformers allow language models to understand relationships between tokens in context.

I learned how self-attention works, why Query, Key, and Value vectors are used, how attention scores are calculated, and how these scores are converted into contextual token representations.

I also implemented the scaled dot-product attention mechanism using NumPy.

---

## Core Concepts

### Why Transformers Matter

Earlier sequence models such as RNNs and LSTMs process information sequentially.

Transformers use attention mechanisms that allow tokens to directly consider relationships with other tokens in the sequence.

Conceptually:

```text
Token 1 ───┐
Token 2 ───┼──→ Attention
Token 3 ───┤
Token 4 ───┘
```

This makes it possible for the model to determine which parts of the context are most relevant when processing each token.

---

## Self-Attention

Self-attention allows tokens in the same sequence to interact with one another.

For example:

```text
The lecturer spoke to the student because she wanted to discuss the assignment.
```

The representation of `she` can incorporate information from other tokens such as `lecturer`.

The important idea is:

```text
Each token
    ↓
Considers other tokens
    ↓
Assigns different attention weights
    ↓
Produces a context-aware representation
```

The word itself does not literally "look" at other words. The process is implemented mathematically using learned vector representations and matrix operations.

---

## Query, Key, and Value

Each token representation is projected into three vectors:

* **Query (Q)** — represents what information the token is looking for.
* **Key (K)** — represents the information available for comparison.
* **Value (V)** — contains the information that can contribute to the resulting representation.

Conceptually:

```text
Query
  ↓
Compare with Keys
  ↓
Calculate relevance
  ↓
Use attention weights
  ↓
Combine Values
```

---

## Scaled Dot-Product Attention

The attention mechanism is represented by:

```text
Attention(Q, K, V)
=
softmax((QKᵀ) / √dₖ)V
```

The process can be broken down into four main steps.
### 1. Calculate Attention Scores

```text
QKᵀ
```

Queries are compared with Keys to determine how strongly tokens relate to one another.


### 2. Scale the Scores


### 3. Apply Softmax

softmax(...)
```
Softmax converts the scores into attention weights.

The weights in each row sum to approximately:
```text
1.0
```
Higher weights indicate a stronger contribution from that token.

---

### 4. Combine the Values

```text
Attention Weights × V
```

The Value vectors are combined according to the calculated attention weights.

The result is a new context-aware representation for each token.

## Multi-Head Attention


Different heads can learn different relationships within the same sequence.

Conceptually:

```text
 │
 ├── Attention Head 1
 ├── Attention Head 2
 ├── Attention Head 3
 └── Attention Head 4
 │
Combine Results
 ↓
Output
```

One head may focus on grammatical relationships while another may capture semantic or positional relationships.

---

## Positional Information

Attention alone does not inherently understand token order.

For example:

```text
AI teaches humans
```

and:

```text
Humans teach AI
```

contain similar tokens but have different meanings.

Transformers therefore include positional information so that token order can influence the model.

Conceptually:

```text
Token Embedding
      +
Position Information
      ↓
Transformer
```

---

## Causal Attention

Autoregressive language models must not use future tokens when predicting the next token.

A causal mask restricts each token so that it can only attend to itself and previous tokens.

Conceptually:

```text
AI          → AI

will        → AI, will

change      → AI, will, change

education   → AI, will, change, education
```

This allows GPT-style models to generate text one token at a time.

---

## Practical Exercise

### `attention_demo.py`

For this lesson, I implemented a simplified attention mechanism using NumPy.

The program:

1. Creates artificial token embeddings.
2. Projects the embeddings into Query, Key, and Value vectors.
3. Calculates raw attention scores.
4. Scales the attention scores.
5. Applies softmax.
6. Produces attention weights.
7. Combines the Value vectors.
8. Displays how much attention each token assigns to the others.

The core calculations are:

```python
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T

scaled_scores = scores / np.sqrt(d_k)

attention_weights = softmax(scaled_scores)

output = attention_weights @ V
```

---

## Experiment

The original example contained three tokens.

I extended it to four tokens:

```text
AI
helps
students
University
```

This changed the attention matrix from:

```text
3 × 3
```

to:

```text
4 × 4
```

because every Query is compared with every Key.

For four tokens:

```text
4 Queries × 4 Keys = 16 attention scores
```

This also demonstrates why standard attention becomes increasingly computationally expensive as sequence length increases.

---

## Interpreting My Attention Results

One of the generated attention rows was:

```text
helps pays attention to:

AI          0.578
helps       0.141
students    0.141
University  0.141
```

This means that, using the artificial matrices in this exercise, the representation for `helps` received the greatest contribution from the `AI` token.

Another result was:

```text
students pays attention to:

AI          0.250
helps       0.250
students    0.250
University  0.250
```

The attention was distributed equally because the underlying attention scores were equal.

These values should not be interpreted as meaningful linguistic relationships because the embeddings and Query, Key, and Value matrices in this exercise were manually created for demonstration purposes.

In a real Transformer, these parameters are learned during model training.

---

## Key Takeaway

The main attention pipeline from this lesson is:

```text
Token Representations
        ↓
Create Q, K, and V
        ↓
Compare Queries and Keys
        ↓
Attention Scores
        ↓
Scale
        ↓
Softmax
        ↓
Attention Weights
        ↓
Weighted Combination of Values
        ↓
Context-Aware Representations
```

Combining Lessons 1 and 2 gives the following simplified LLM pipeline:

```text
Prompt
 ↓
Tokenisation
 ↓
Token IDs
 ↓
Embeddings
 ↓
Positional Information
 ↓
Transformer Layers
 ↓
Self-Attention
 ↓
Context-Aware Representations
 ↓
Next-Token Probabilities
 ↓
Generated Token
 ↓
Repeat
```

---

## Files

* `README.md` — Summary of Transformer and attention concepts.
* `attention_demo.py` — NumPy implementation of scaled dot-product attention.

---

## Status

**Lesson 2 completed.**
 ↓
Input
Transformers normally use several attention heads rather than only one.

---



```text
```text
QKᵀ / √dₖ
```
The scores are divided by the square root of the Key dimension to prevent very large values from making the softmax distribution unstable.

---

---

