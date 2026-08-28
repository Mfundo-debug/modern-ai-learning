from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# --------------------------------------------------
# STEP 1: CREATE A SMALL KNOWLEDGE BASE
# --------------------------------------------------

knowledge_base = """
Retrieval-Augmented Generation, commonly called RAG, combines
information retrieval with a generative language model. Instead
of relying only on information stored in the model's parameters,
a RAG system retrieves relevant external information before
generating an answer.

Embeddings are numerical vector representations of information.
Text with similar meaning tends to have related embedding
representations. Embeddings are commonly used in semantic search
because they allow queries and documents to be compared according
to meaning rather than only exact keyword matches.

A RAG system normally begins by dividing large documents into
smaller pieces called chunks. Each chunk is converted into an
embedding and stored for later retrieval. When a user asks a
question, the question is also embedded and compared with the
stored document embeddings.

Chunk overlap can help preserve information that occurs near a
chunk boundary. Without overlap, an important sentence or concept
may be divided between two chunks, reducing the quality of
retrieval.

After retrieval, the most relevant chunks are placed into the
language model's context together with the user's question. The
language model is instructed to answer using the retrieved
information. This process is known as grounding.

RAG can reduce hallucination by giving the language model relevant
external evidence, but it does not guarantee a correct answer.
The retrieval system may retrieve irrelevant information, or the
language model may incorrectly interpret the retrieved context.

Top-k retrieval means selecting several of the highest-ranking
document chunks instead of only one. Retrieving multiple chunks
can be useful when the information required to answer a question
is distributed across different parts of a document.

A vector store is a system designed to store, index, and search
vector representations efficiently. Examples of vector retrieval
technologies include FAISS, pgvector, Pinecone, Chroma, and
Weaviate.
"""


# --------------------------------------------------
# STEP 2: SPLIT THE KNOWLEDGE BASE INTO CHUNKS
# --------------------------------------------------

def chunk_text(text, chunk_size=60, overlap=15):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        chunks.append(" ".join(chunk))

        if end >= len(words):
            break

        start = end - overlap

    return chunks


chunks = chunk_text(knowledge_base)


print("NUMBER OF CHUNKS:")
print(len(chunks))


# --------------------------------------------------
# STEP 3: LOAD THE EMBEDDING MODEL
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# STEP 4: EMBED THE CHUNKS
# --------------------------------------------------

chunk_embeddings = embedding_model.encode_document(
    chunks
)


print("\nEMBEDDING SHAPE:")
print(chunk_embeddings.shape)


# --------------------------------------------------
# STEP 5: GET A QUESTION
# --------------------------------------------------

question = input("\nAsk a question about RAG: ")


# --------------------------------------------------
# STEP 6: EMBED THE QUESTION
# --------------------------------------------------

question_embedding = embedding_model.encode_query(
    [question]
)


# --------------------------------------------------
# STEP 7: CALCULATE SIMILARITY
# --------------------------------------------------

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]


# --------------------------------------------------
# STEP 8: CHECK RETRIEVAL CONFIDENCE
# --------------------------------------------------

retrieval_threshold = 0.30

best_score = similarities.max()

print("\nBEST RETRIEVAL SCORE:")
print(round(float(best_score), 3))


if best_score < retrieval_threshold:

    print("\nRAG ANSWER:\n")
    print(
        "I do not have enough information "
        "in the knowledge base."
    )

else:

    # --------------------------------------------------
    # STEP 9: RETRIEVE TOP-K CHUNKS
    # --------------------------------------------------

    top_k = 2

    top_indices = similarities.argsort()[::-1][:top_k]

    retrieved_chunks = []

    print("\nRETRIEVED CHUNKS:\n")

    for index in top_indices:

        chunk = chunks[index]
        score = similarities[index]

        retrieved_chunks.append(chunk)

        print(f"Similarity: {score:.3f}")
        print(chunk)
        print()


    # --------------------------------------------------
    # STEP 10: BUILD THE CONTEXT
    # --------------------------------------------------

    context = "\n\n".join(retrieved_chunks)


    # --------------------------------------------------
    # STEP 11: LOAD LOCAL GENERATION MODEL
    # --------------------------------------------------

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    generator_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )
    )


    # --------------------------------------------------
    # STEP 12: BUILD THE RAG PROMPT
    # --------------------------------------------------

    prompt = f"""
    Answer the question using only the information
    contained in the context below.

    Do not use outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """


    # --------------------------------------------------
    # STEP 13: GENERATE THE ANSWER
    # --------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    outputs = generator_model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nRAG ANSWER:\n")
    print(answer)