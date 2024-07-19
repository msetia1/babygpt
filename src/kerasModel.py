import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import random
import re


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,'input.txt')

# Function to help filter out speaker tags from input data
def is_speaker_tag(line):
    return re.match(r'^([A-Z\s]+|([A-Z][a-z]+(\s[A-Z][a-z]+)*)):$', line)

with open(file_path, 'r', encoding = 'utf-8') as file:
    input = file.read().lower()

# Breaks up input into lines
lines = input.split('\n')
text_parts = []
dialogue_buffer = []

for line in lines:
    # Gets rid of whitespace in line
    stripped_line = line.strip()
    if not stripped_line:
        continue  # Skip empty lines

    if is_speaker_tag(stripped_line):
        if dialogue_buffer:
            # Append buffered dialogue lines to text_parts
            text_parts.append(' '.join(dialogue_buffer))
            dialogue_buffer = []
    else:
        dialogue_buffer.append(stripped_line)  # Buffer the dialogue line

# Append any remaining buffered dialogue
if dialogue_buffer:
    text_parts.append(' '.join(dialogue_buffer))

# Combine all text parts into a single string
text = ' '.join(text_parts)


# Create a set of all unique chars in order to create integer to char encoding and vice versa
chars = sorted(set(text))
char_indices = {c : i for i, c in enumerate(chars)}
indices_char = {i : c for i, c in enumerate(chars)}

# Sequence length subject to change
sequence_length = 40
input_sequences = []
output_sequences = []

step_size = 3
# Create input sequence for context for model to learn patterns and output sequence for target prediction
# Step size decreases training time
for i in range(0, len(text) - sequence_length, step_size):
    input_sequence = text[i:i + sequence_length]
    output_sequence = text[i + sequence_length]
    input_sequences.append([char_indices[char] for char in input_sequence])
    output_sequences.append(char_indices[output_sequence])

# convert to numpy arrays to make manipulation easier
input_sequences = np.array(input_sequences)
output_sequences = np.array(output_sequences)

# Create training input and output data
num_chars = len(chars)
X = np.zeros((len(input_sequences), sequence_length, num_chars), dtype='bool')
y = np.zeros((len(output_sequences), num_chars), dtype='bool')

# Convert input and output sequences into matrices of 0s and 1s
for i, input_sequence in enumerate(input_sequences):
    X[i] = keras.utils.to_categorical(input_sequence, num_classes=num_chars)
    y[i] = keras.utils.to_categorical(output_sequences[i], num_classes=num_chars)

# Softmax is best activation function for the output layer for a char level gen because 
# it provides a distribution for the next possible characters
model = keras.Sequential([
    keras.Input(shape = (sequence_length, num_chars)),
    layers.LSTM(units = 128, activation = 'relu'),
    layers.Dropout(rate = 0.2),
    layers.Dense(units = len(chars), activation = 'softmax')
])

optimizer = keras.optimizers.AdamW(learning_rate = 0.001)

model.compile(
    optimizer = optimizer,
    loss = 'categorical_crossentropy'
)

# Text sampling function that creates a sampling distribution of characters 
# and returns the character with the highest probability
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Epoch is the number of times the model will go through the training data
epochs = 40
# Batch size is the number of samples from the data that is processed before the model is updated
batch_size = 128

# Iterating through each epoch
for epoch in range(epochs):
    # Updating model each iteration
    model.fit(X, y, batch_size = batch_size, epochs = 1)
    print()
    print("Generating text after epoch: %d" % epoch)

    # Random start index for sentence in data
    start_index = random.randint(0, len(text) - sequence_length - 1)
    # Diversity affects randomness of character generation
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print("...Diversity:", diversity)

        generated = ""
        sentence = text[start_index : start_index + sequence_length]
        print('...Generating with seed: "' + sentence + '"')

        for i in range(400):
            x_pred = np.zeros((1, sequence_length, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.0
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            sentence = sentence[1:] + next_char
            generated += next_char

        print("...Generated: ", generated)
        print("-")

