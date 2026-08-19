from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# STEP 1: CREATE A SMALL KNOWLEDGE BASE
# --------------------------------------------------

documents = [
    "Python for loops are used to repeat a block of code over a sequence.",

    "A neural network is a machine learning model made up of interconnected layers of artificial neurons.",

    "Cloud computing provides computing resources such as storage and processing power over the internet.",

    "A database stores organised information that can be searched, updated, and managed.",

    "Retrieval-Augmented Generation retrieves relevant external information before generating an LLM response.",

    "Self-attention allows Transformer models to determine which tokens are most relevant to one another.",

    "Supervised machine learning trains models using labelled examples.",

    "A firewall helps control network traffic according to predefined security rules."
]


# ----------------------------------------------------
# STEP 2: LOAD AN EMBEDDDING MODEL
# ----------------------------------------------------

model = SentenceTransformer(

	"sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------
# STEP 3: EMBED THE DOCUMENTS
# ---------------------------------------------------

document_embeddings = model.encode(documents)

print("NUMBER OF DOCUMENTS:")
print(len(documents))

print("\nEMBEDDING SHAPE:")
print(document_embeddings.shape)

# ---------------------------------------------------
# STEP 4: GET A SEARCH QUERY
# ---------------------------------------------------

query = input("\nEnter your search query: ")


# --------------------------------------------------
# STEP 5: EMBED THE QUERY
# --------------------------------------------------

query_embedding = model.encode([query])


# --------------------------------------------------
# STEP 6: CALCULATE COSINE SIMILARITIES
# --------------------------------------------------

similarities = cosine_similarity(
query_embedding,
document_embeddings
)[0]

# --------------------------------------------------
# STEP 7: DISPLAY ALL SIMILARITY SCORES
# --------------------------------------------------

print("\nSIMILARITY SCORES:\n")

for document, score in zip(documents, similarities):
    
    print(f"{score:.3f} ->{document}")

# --------------------------------------------------
# STEP 8: FIND THE MOST SIMILAR DOCUMENT
# --------------------------------------------------

best_match_index = similarities.argmax()

best_document = documents[best_match_index]

best_score = similarities[best_match_index]

print("\nBEST MATCH:")
print(best_document)

print("\nSIMILARITY SCORE:")
print(round(float(best_score),3))

# --------------------------------------------------
# STEP 9: RETRIEVE THE TOP 3 RESULTS
# --------------------------------------------------

top_k = 3

top_indices = similarities.argsort()[::-1][:top_k]


print("\nTOP 3 MATCHES:\n")

for index in top_indices:

    print(
        f"{similarities[index]:.3f}"
        f" -> {documents[index]}"
    )
