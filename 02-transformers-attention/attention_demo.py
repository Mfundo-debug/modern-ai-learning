import numpy as np


# --------------------------------------------------
# STEP 1: CREATE SIMPLE TOKEN REPRESENTATIONS
# --------------------------------------------------

tokens = ["AI", "helps", "students", "University"]

# Each token has a very small artificial embedding.
# Real LLM embeddings contain hundreds or thousands
# of dimensions.

X = np.array([
    [1.0, 0.0, 1.0],   # AI
    [0.0, 2.0, 1.0],   # helps
    [1.0, 1.0, 0.0],    # students
    [1.0, 1.0, 0.0]   # University
])


print("TOKENS:")
print(tokens)

print("\nTOKEN REPRESENTATIONS:")
print(X)


# --------------------------------------------------
# STEP 2: DEFINE QUERY, KEY AND VALUE WEIGHTS
# --------------------------------------------------

# In a real Transformer these matrices are learned
# during training.

W_Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

W_K = np.array([
    [1.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

W_V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])


# --------------------------------------------------
# STEP 3: CALCULATE Q, K AND V
# --------------------------------------------------

Q = X @ W_Q
K = X @ W_K
V = X @ W_V


print("\nQUERIES:")
print(Q)

print("\nKEYS:")
print(K)

print("\nVALUES:")
print(V)


# --------------------------------------------------
# STEP 4: CALCULATE ATTENTION SCORES
# --------------------------------------------------

scores = Q @ K.T

print("\nRAW ATTENTION SCORES:")
print(scores)


# --------------------------------------------------
# STEP 5: SCALE THE SCORES
# --------------------------------------------------

d_k = K.shape[1]

scaled_scores = scores / np.sqrt(d_k)

print("\nSCALED ATTENTION SCORES:")
print(scaled_scores)


# --------------------------------------------------
# STEP 6: SOFTMAX
# --------------------------------------------------

def softmax(matrix):

    exp_values = np.exp(
        matrix - np.max(matrix, axis=1, keepdims=True)
    )

    return exp_values / np.sum(
        exp_values,
        axis=1,
        keepdims=True
    )


attention_weights = softmax(scaled_scores)

print("\nATTENTION WEIGHTS:")
print(attention_weights)


# --------------------------------------------------
# STEP 7: COMBINE THE VALUES
# --------------------------------------------------

output = attention_weights @ V

print("\nATTENTION OUTPUT:")
print(output)


# --------------------------------------------------
# STEP 8: DISPLAY ATTENTION CLEARLY
# --------------------------------------------------

print("\nATTENTION BY TOKEN:")

for i, token in enumerate(tokens):

    print(f"\n{token} pays attention to:")

    for j, other_token in enumerate(tokens):

        print(
            f"  {other_token}: "
            f"{attention_weights[i][j]:.3f}"
        )