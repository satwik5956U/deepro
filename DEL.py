import pandas as pd

# Define the file path to your Excel sheet
excel_file_path = "C:/Users/NEELAM/OneDrive/Desktop/deepro/English-Telugu.csv"

# Read data from the Excel sheet into DataFrames
df = pd.read_csv(excel_file_path)

# Access specific columns from the DataFrame
source_column_name = "English"
target_column_name = "Telugu"

source_sentences = df[source_column_name].tolist()
target_sentences = df[target_column_name].tolist()

# Print a few examples to verify data loading
print("Source Sentences:")
ss=[]
tt=[]
for i in range(len(source_sentences)):
    print(source_sentences[i])
    ss.append(list(source_sentences[i].split(' ')))

print("Target Sentences:")
for i in range(5):
    print(target_sentences[i])
    tt.append(list(target_sentences[i].split(' ')))
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Example data (tokenized)
source_sentences = ss
target_sentences = tt

# Vocabulary
source_vocab = set(word for sentence in source_sentences for word in sentence)
target_vocab = set(word for sentence in target_sentences for word in sentence)
source_vocab_size = len(source_vocab)
target_vocab_size = len(target_vocab)

# Create word-to-index and index-to-word dictionaries
source_word2idx = {word: idx for idx, word in enumerate(source_vocab)}
source_idx2word = {idx: word for word, idx in source_word2idx.items()}
target_word2idx = {word: idx for idx, word in enumerate(target_vocab)}
target_idx2word = {idx: word for word, idx in target_word2idx.items()}

# Maximum sequence lengths
max_source_seq_length = max(len(sentence) for sentence in source_sentences)
max_target_seq_length = max(len(sentence) for sentence in target_sentences)

# Convert sentences to numerical data
source_sequences = [[source_word2idx[word] for word in sentence] for sentence in source_sentences]
target_sequences = [[target_word2idx[word] for word in sentence] for sentence in target_sentences]

# Pad sequences
source_sequences = tf.keras.preprocessing.sequence.pad_sequences(source_sequences, maxlen=max_source_seq_length)
target_sequences = tf.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_target_seq_length)

# Define the RNN model
embedding_dim = 128
hidden_units = 256

# Encoder
encoder_inputs = Input(shape=(max_source_seq_length,))
encoder_embedding = Embedding(input_dim=source_vocab_size, output_dim=embedding_dim)(encoder_inputs)
encoder = LSTM(hidden_units, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_embedding)

# Decoder
decoder_inputs = Input(shape=(max_target_seq_length,))
decoder_embedding = Embedding(input_dim=target_vocab_size, output_dim=embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = Dense(target_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()
model_save_path = "translation_model.h5"

# Save the model to an HDF5 file
model.save(model_save_path)

# Print a message to confirm that the model has been saved
print(f"Model saved to {model_save_path}")
# Train the model
# For a complete training loop, you would need to set up data generators and train the model over multiple epochs.

# Inference can be implemented for translating text using this trained model.

