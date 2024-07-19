# BabyGPT: A Character-Level Generation Model built with Keras
BabyGPT is a term for language models similar to that of ChatGPT with the main difference being the size of the training data.\
ChatGPT is a Large Language Model (LLM) that is trained on a very large dataset, hence the Large in LLM.\
BabyGPT on the other hand is a similar type of language model except the training data used is much smaller, hence the name BabyGPT.

## TLDR
This model takes in 40 characters of text from input text data containing 40,000 lines of Shakespeare's works and tries to correctly predict the following 400 characters.

<br>

## Hyperparameters

The model's performance can be changed by adjusting the following hyperparameters

- `sequence_length`: The length of the input sequence provided to the model as context 
- `units`: The number of neurons in a layer within the neural network
- `activation`: The function that introduces non-linearity into the output of a neuron (eg: ReLU, sigmoid, TanH, etc.)
- `optimizer`: The function that adjusts the attributes, like weights and biases, of the neural network in order to reduce loss (eg: Adam, RMSProp, SGD, etc.)
- `learning_rate`: Determines how much the model's weights change in response to the loss
- `loss`: The function used to calculate the loss of the model, loss being the difference bewteen the model's predictions and the actual target values (eg: categorical crossentropy, MSE, etc.)
- `epochs`: The number of passes of the entire training dataset passing through the model
- `batch_size`: The number of samples that is fed into the model at each iteration of the training process

The current values/choices for these parameters was for the most part arbitrarily chosen. They can be adjusted/changed to get a lower loss value.

<br>

## Constructing the Model
### Step 1: Data Preparation
The first step for any machine learning model is to prepare and clean the data so that the model can process it.
For this model, the input data is in the format as follows:\
\
**_Speaker Name:_**\
**_Dialogue_**\
\
However, specific names were in all caps, and generic names, such as Person 1, only had the first letter capitalized
Because of this, I had to implement a function `is_speaker_tag` in order to discern what is and isn't a speaker tag.\
The data preprocessing takes place in lines 13 - 44. It essentially consists of stripping the lines of any whitespaces, scrapping the speaker tags since they are irrelevant to my goal for the model, and then appending all the text to one long string, which is better for character-level generation since you can iterate through each character in a string, like elements in an array.\
\
From there I had to create the character-to-index and index-to-character mapping. This is necessary because models only understand numbers and not characters.\
Then I created lists of input and output sequences, input sequences being the context for the model and the output sequence being the target character that the model is trying to predict

### Step 2: Model Construction
I chose to use only a 3 layer neural network, consisting of an input layer, an LSTM (Long Short-Term Memory) layer, and a Dense layer.\
Layer dropout drops a certain percent of neurons during training to improve generalization and prevent overfitting, so I also included a 20% layer dropout. The number of layers,type of layers, number of neurons, activation functions, etc. can all be changed to try and improve the model.

### Step 3: Training the Model
The text sampling function and training loop were taken from the Keras website directly (see reference below). I adjusted them slightly to work with my specific model. The text sampling function takes the predictions and essentially turns it into a sampling distribution and returns the character index with the highest probability of being the next character.

## References Used
- Article that sparked the idea: https://www.nytimes.com/interactive/2023/04/26/upshot/gpt-from-scratch.html
- Repository providing input data and some help: https://github.com/TatevKaren/BabyGPT-Build_GPT_From_Scratch/tree/main
- Keras example used for text sampling function and training loop:  https://keras.io/examples/generative/lstm_character_level_text_generation/

