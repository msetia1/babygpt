# BabyGPT: A Character-Level Generation Model built with Keras
BabyGPT is a term for language models similar to that of ChatGPT with the main difference being the size of the training data.\
ChatGPT is a Large Language Model (LLM) that is trained on a very large dataset, hence the Large in LLM.\
BabyGPT on the other hand is a similar type of language model except the training data used is much smaller, hence the name BabyGPT.

## TLDR
This model takes in 40 characters of text from input text data containing 40,000 lines of Shakespeare's works and tries to correctly predict the following 400 characters.

## Sample Output
``` console
2886/2886 ━━━━━━━━━━━━━━━━━━━━ 88s 30ms/step - loss: 2.6403   

Generating text after epoch: 0
...Diversity: 0.2
...Generating with seed: "not weep; for all my body's moisture sca"
...Generated:  nd the sand the serst and the sure the bate the shave the buther sore and the the the shave the pare the sore the sore the sore the serent and the sere the sore the sore the seres and and the sere the sordent and the sore the bucher the bather sore the pare the beather sore the sall and the the wish the bucher to the sores and the conders me the come the buther and the the mand and and the the sor
-
...Diversity: 0.5
...Generating with seed: "not weep; for all my body's moisture sca"
...Generated:  ringe and in ald i wer the mangst if a doond for the kerist and mishers vavence to works and hat me sing to cumens coants of her sicon the bander i and sore fore of my and the sead in wiss mene the beast have that i seat i ow the dather the erines and buther seed in youl the pearout come in the soule and of the sorverent cour the cavers and the forst thes sit and and and then and and davend and i 
-
...Diversity: 1.0
...Generating with seed: "not weep; for all my body's moisture sca"
...Generated:  menow. arbuk: gaventersang: thom thathe'te dilettess forshageres the sereel-heng dorl thee! froi ga wo damnine proth yourst race teather shiverosce coring vef theeptt! i butwild; like in fid netehare sceruthagld: of m ponees hal be bol, the suandred fpuncais conberunpy pliop thes nam sif wouks, whac angind. whey thin if it ho look: wett, buce prerelt: of menes: of seandy of umant: if i is rithalls
-
...Diversity: 1.2
...Generating with seed: "not weep; for all my body's moisture sca"
...Generated:  s. wher diest ghave if gias? draed'n r vontan thimadbe houcesgor eamn ther thes lites, thy shaveg mo houeys nocrory sotk;-hire a frrint: i bi sirf of wlave satteet. keds owss to will! srendit. muktenpetf bppock wisuspilingxsnef dutn bryown tu cente prie, seal why? i the skand: gente thimeres'le. my or; botki, non, norstloug; aghird inom: winp ast urean'st you whik e wnak, rilallin:  hoo banneln he
```
...
``` console
2886/2886 ━━━━━━━━━━━━━━━━━━━━ 44095s 15s/step - loss: 1.4411  

Generating text after epoch: 39
...Diversity: 0.2
...Generating with seed: " earth which serves as paste and cover t"
...Generated:  o the come of the heaven to his house the counterfect and the counterfect be the weak me the counterfelless and the countertion of the fair of the courtes and for the dear the charged and stand the prince of the stread of the bearth of the come, and the ward the liest with the world that with the precenter to the state to hear the counterfect and the state the entleased to his heart of the sure to
-
...Diversity: 0.5
...Generating with seed: " earth which serves as paste and cover t"
...Generated:  o this death. northumberland: sir, the king the mather as the courtes of what you an of my cearles like the softing and the store to the spired the speep of the content of book and you have death to accuse the provertion. the shall would be and the forth it is that seem him to my son have deach the good to the life, be and which we have shall i will is a mut to the maid down, for the was i should 
-
...Diversity: 1.0
...Generating with seed: " earth which serves as paste and cover t"
...Generated:  o cure. o cry'll: the noble corsomen morrow like a prayer are it off, for, mercy down he houch he that good cry nor dompion and beouthed. this shit an atter, madam let so? who? what shall breaths away, go of thee we the for a jeading, will creasu's of into us. the enemers will not will his briaghs, i welcomious behold. clarence: in tarks; prefin who have gone; before thy feast in it of this contio
-
...Diversity: 1.2
...Generating with seed: " earth which serves as paste and cover t"
...Generated:  o the drlouter of deid, cith rurces serams is traught duchtes povertatness of his quock humbal. i, caley liper hopewars for me but frant, laids some jester of thy uney:-and. courd end gincled! freait, romeo: gloucester, there be 'umbly add being oon the blood; that was staid up, who strivestance? northumberland: lid: is, and hath, here not, pardoada regold dean. juliet: how case you for, what mudy
```

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
- `diversity`: The randomness of the selection for the next character choice; lower = less random, higher = more random

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

