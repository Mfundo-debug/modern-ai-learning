import random
from collections import defaultdict, Counter


text = """
artificial intelligence is changing technology
artificial intelligence is changing education
artificial intelligence is changing healthcare
machine learning is part of artificial intelligence
deep learning is part of machine learning
generative ai can generate text
generative ai can generate images
generative ai can generate code
ai agents can use tools
ai agents can perform tasks
"""


# --------------------------------------------------
# STEP 1: TOKENISE THE TEXT
# --------------------------------------------------

tokens = text.lower().split()

print("TOKENS:")
print(tokens)
print()


# --------------------------------------------------
# STEP 2: LEARN WHICH WORDS FOLLOW OTHER WORDS
# --------------------------------------------------

transitions = defaultdict(Counter)

for current_word, next_word in zip(tokens, tokens[1:]):
    transitions[current_word][next_word] += 1


print("WORDS THAT CAN FOLLOW 'artificial':")
print(transitions["artificial"])
print()


print("WORDS THAT CAN FOLLOW 'generate':")
print(transitions["generate"])
print()


# --------------------------------------------------
# STEP 3: CONVERT COUNTS TO PROBABILITIES
# --------------------------------------------------

def get_probabilities(word):

    possible_words = transitions[word]

    total = sum(possible_words.values())

    probabilities = {}

    for next_word, count in possible_words.items():
        probabilities[next_word] = count / total

    return probabilities


print("PROBABILITIES AFTER 'generate':")

probabilities = get_probabilities("generate")

for word, probability in probabilities.items():
    print(word, round(probability, 3))

print()


# --------------------------------------------------
# STEP 4: SAMPLE THE NEXT WORD
# --------------------------------------------------

def predict_next_word(word):

    possible_words = transitions[word]

    words = list(possible_words.keys())
    weights = list(possible_words.values())

    return random.choices(
        words,
        weights=weights,
        k=1
    )[0]


print("PREDICTED WORD AFTER 'generate':")
print(predict_next_word("generate"))
print()


# --------------------------------------------------
# STEP 5: GENERATE A SEQUENCE
# --------------------------------------------------

def generate_text(start_word, length=10):

    current_word = start_word

    generated_words = [current_word]

    for _ in range(length - 1):

        if current_word not in transitions:
            break

        next_word = predict_next_word(current_word)

        generated_words.append(next_word)

        current_word = next_word

    return " ".join(generated_words)


print("GENERATED TEXT:")
print(generate_text("artificial", 12))
