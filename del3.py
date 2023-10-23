import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("translation_model.h5")

# Define dictionaries for word-to-index and index-to-word mappings
# These should be consistent with the mappings used during training
source_word2idx = {"I": 0, "love": 1, "AI": 2}
target_word2idx = {"J'aime": 0, "l'IA": 1, "Translate": 2, "this": 3, "sentence": 4, "Traduis": 5, "cette": 6, "phrase": 7}

# Function to translate a source sentence
def translate_sentence(source_sentence):
    # Tokenize and convert the source sentence to numerical data
    source_sequence = [source_word2idx[word] for word in source_sentence.split()]

    # Initialize the target sequence with a start token
    target_sequence = [target_word2idx["<start>"]]

    # Maximum sequence length for the target language
    max_target_seq_length = 10  # You can adjust this based on your training data

    # Perform translation
    for _ in range(max_target_seq_length):
        # Predict the next word in the target sequence
        predicted_probs = loaded_model.predict([np.array([source_sequence]), np.array([target_sequence])])

        # Get the index of the most likely next word
        next_word_idx = np.argmax(predicted_probs[0, -1, :])

        # Append the next word to the target sequence
        target_sequence.append(next_word_idx)

        # If the next word is the end token, stop
        if next_word_idx == target_word2idx["<end>"]:
            break

    # Convert the numerical target sequence back to words
    target_sentence = " ".join(target_idx2word[idx] for idx in target_sequence[1:-1])  # Exclude start and end tokens

    return target_sentence

# User input
user_input = "I love AI"

# Translate the user input
translated_output = translate_sentence(user_input)

print("Input Sentence:", user_input)
print("Translated Sentence:", translated_output)
