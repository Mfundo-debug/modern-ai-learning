# Lesson 5: Embeddings and Semantic Search

## What I Learned

This lesson introduced embeddings and semantic search as the foundation for modern retrieval systems and Retrieval-Augmented Generation (RAG).

I learned how text can be converted into numerical vectors, how cosine similarity is used to compare those vectors, and how semantic search retrieves documents based on meaning rather than exact keyword overlap.

I also observed an important retrieval failure: a very small change in query wording changed the ranking of retrieved documents. This demonstrated that embeddings approximate meaning rather than understanding it perfectly.

---

## Core Concepts

### Embeddings

An embedding is a numerical vector representation of information.

For example:

```text
"machine learning"
        ↓
Embedding Model
        ↓
[0.12, -0.43, 0.87, ...]
```

The individual dimensions do not usually have simple human-readable meanings.

Instead, the overall position of the vector captures useful information about the input.

Embeddings can represent:

* Words
* Sentences
* Paragraphs
* Document chunks
* Questions
* Code
* Images

---

## Semantic Similarity

Semantically related pieces of text tend to have embedding vectors that are closer to one another.

Conceptually:

```text
Machine Learning
        ●
     ● Deep Learning


                         ● Database


● Banana
```

This makes it possible to compare text based on meaning rather than exact wording.

---

## Keyword Search vs Semantic Search

Keyword search relies heavily on exact or overlapping words.

For example:

```text
Query:
How do I repeat code in Python?
```

A relevant document may say:

```text
Iteration allows instructions to execute repeatedly.
```

The wording is different, but the meaning is related.

Semantic search can still identify this relationship by comparing embeddings.

The process is:

```text
Query
 ↓
Embedding
 ↓
Compare with document embeddings
 ↓
Similarity scores
 ↓
Rank documents
```

---

## Cosine Similarity

Cosine similarity compares the direction of two vectors.

The formula is:

```text
cosine_similarity(A, B)
=
(A · B) / (||A|| ||B||)
```

Conceptually:

```text
Small angle between vectors
        ↓
Higher similarity
```

and:

```text
Large angle between vectors
        ↓
Lower similarity
```

Cosine similarity is commonly used for comparing embeddings.

---

## Important Similarity Principle

A higher similarity score means that one result is more similar than another according to the embedding model.

It does not automatically mean that the result is correct.

For example:

```text
Document A → 0.42
Document B → 0.31
Document C → 0.18
```

Document A is the highest-ranked result.

However:

```text
highest similarity
        ≠
guaranteed relevance
```

Similarity is a ranking signal, not a truth guarantee.

---

## Practical Exercise

### `semantic_search.py`

For this lesson, I built a small semantic search engine using:

* Sentence Transformers
* `all-MiniLM-L6-v2`
* Scikit-learn
* Cosine similarity

The system contains a small knowledge base with documents about topics such as:

* Python loops
* Neural networks
* Cloud computing
* Databases
* RAG
* Self-attention
* Supervised learning
* Firewalls

The retrieval pipeline is:

```text
Documents
   ↓
Embedding Model
   ↓
Document Embeddings
   ↓

User Query
   ↓
Embedding Model
   ↓
Query Embedding
   ↓

Cosine Similarity
   ↓
Rank Documents
   ↓
Best Matches
```

---

## Embedding Dimensions

The embedding model used in this exercise produces 384-dimensional vectors.

With 8 documents, the embedding matrix had the shape:

```text
(8, 384)
```

This means:

```text
8 documents
×
384 embedding dimensions
```

Each document is represented by its own 384-dimensional vector.

---

## Basic Semantic Search Implementation

The core retrieval logic is:

```python
document_embeddings = model.encode(documents)

query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]
```

The highest-scoring document can then be identified using:

```python
best_match_index = similarities.argmax()
```

---

## Top-k Retrieval

Instead of retrieving only one document, a search system can retrieve several high-ranking results.

For example:

```python
top_k = 3

top_indices = similarities.argsort()[::-1][:top_k]
```

This produces the top three documents according to their similarity scores.

Conceptually:

```text
Query
 ↓
Embedding
 ↓
Similarity Search
 ↓
Rank Documents
 ↓
Top 3 Results
```

Top-k retrieval becomes especially important in RAG because the information required to answer a question may be spread across several document chunks.

---

## Important Experiment: Query Sensitivity

One of the most interesting results from this lesson occurred when testing the query:

```text
How can AI system use information outside its training data?
```

The expected semantic match was the RAG document:

```text
Retrieval-Augmented Generation retrieves relevant external information before generating an LLM response.
```

However, the highest-scoring result was:

```text
A neural network is a machine learning model made up of interconnected layers of artificial neurons.
```

with a similarity score of approximately:

```text
0.418
```

while the RAG document scored approximately:

```text
0.286
```

This was a retrieval failure.

---

## Why the Retrieval Failed

The query contained concepts such as:

```text
AI
system
training data
```

The neural-network document contained strongly related machine-learning terminology.

The RAG document instead contained:

```text
retrieval
external information
LLM response
```

A human can understand that:

```text
using information outside training data
```

is closely related to RAG.

However, the embedding model did not represent that relationship strongly enough for this particular query and document wording.

This demonstrated an important principle:

```text
semantic similarity
        ≠
perfect semantic understanding
```

---

## Small Wording Changes Can Affect Retrieval

A very small change in query wording was also observed to affect the retrieved result.

For example:

```text
How can an AI system use information outside its training data?
```

and:

```text
How can AI system use information outside its training data?
```

are almost identical to a human reader.

However, they produce different token sequences and therefore slightly different embedding vectors.

Conceptually:

```text
Query wording
      ↓
Embedding changes
      ↓
Similarity scores change
      ↓
Document ranking can change
```

This does not mean that words such as `an` are inherently important.

The important lesson is that embedding-based retrieval is sensitive to the representation produced from the full input sequence.

---

## Document Wording Also Matters

Retrieval quality depends not only on the user query but also on how documents are written and chunked.

The original RAG document was:

```text
Retrieval-Augmented Generation retrieves relevant external information before generating an LLM response.
```

A richer version could be:

```text
Retrieval-Augmented Generation allows an AI or language model to use information outside its training data by retrieving relevant external documents before generating a response.
```

This wording is more closely aligned with queries about external information and training data.

This demonstrates that:

```text
better document representation
        ↓
better retrieval
```

---

## Retrieval Failure Is Different From LLM Failure

This lesson also introduced an important RAG engineering principle.

Consider:

```text
User Question
      ↓
Wrong Document Retrieved
      ↓
Wrong Context Given to LLM
      ↓
Poor Final Answer
```

The LLM may appear to have failed.

But the actual failure occurred earlier:

```text
retrieval failed
```

This means RAG systems should be evaluated at different stages rather than treating the entire application as one black box.

---

## Vector Stores

In this lesson, document embeddings were stored directly in memory.

That works for a very small collection.

For a large system, specialised vector stores or indexes are used to store and search embeddings efficiently.

Examples include:

* FAISS
* Chroma
* Pinecone
* Weaviate
* pgvector

The responsibilities are different:

```text
Embedding Model
       ↓
Creates vectors

Vector Store
       ↓
Stores, indexes, and searches vectors
```

An embedding model is therefore not the same thing as a vector database.

---

## Embeddings vs Semantic Search

These concepts should also not be confused.

```text
Embedding
=
numerical representation
```

while:

```text
Semantic Search
=
using representations to retrieve information based on meaning
```

A useful mental model is:

```text
Semantic Embeddings
        +
Vector Similarity Search
        ↓
Semantic Search
```

---

## Connection to RAG

This lesson built the retrieval component needed for RAG.

Current pipeline:

```text
User Question
      ↓
Create Query Embedding
      ↓
Compare with Document Embeddings
      ↓
Retrieve Relevant Documents
```

RAG adds another stage:

```text
Relevant Documents
      +
User Question
      ↓
LLM
      ↓
Grounded Answer
```

The complete simplified RAG architecture is therefore:

```text
User Question
      ↓
Embedding Model
      ↓
Semantic Search
      ↓
Retrieve Relevant Context
      ↓
LLM
      ↓
Generated Answer
```

---

## Key Takeaways

The main retrieval pipeline from this lesson is:

```text
Text
 ↓
Embedding Model
 ↓
Vector Representation
 ↓
Cosine Similarity
 ↓
Document Ranking
 ↓
Semantic Retrieval
```

The most important engineering lesson was:

```text
Highest similarity
        ≠
Correct retrieval
```

Retrieval quality depends on:

* Query wording
* Document wording
* Embedding model quality
* Chunking strategy
* Similarity method
* Retrieval size
* Domain
* Evaluation

Small changes in the input can affect rankings, and a retrieval system can confidently return the wrong document.

This means that retrieval must be evaluated separately from the LLM that eventually consumes the retrieved context.

---

## Files

* `README.md` — Summary of embeddings, semantic search, and retrieval failure observations.
* `semantic_search.py` — Local semantic search implementation using Sentence Transformers and cosine similarity.

---

## Status

**Lesson 5 completed.**
