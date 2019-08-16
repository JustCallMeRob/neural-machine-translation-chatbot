# Neural machine translation chatbot

Applying neural machine translation for the purposes of creating an artificial conversational entity. 
The purpose of this project is to demonstrate the feasibility and implement methods of creating
artificial conversational entities, or chatterbots using a novel new sequence-to-sequence techniques derived 
from advancements in machine learning, specifically in the fields of natural language processing and 
neural machine translation (NMT).

An NMT will be trained on a dataset of comments from Reddit provided by Jason? i think, from [this link](https://files.pushshift.io/reddit/comments/). Now this repository contains every publicly available comment posted to reddit since 2005, i mostly used three months from 2015. The idea being that if i feed comment and response pairs to the NMT it would map simmilar responses to their comment parents and be able to spit out sensible responses for any input.

# Functional Description

## 1. Database creation
Taking the data out of the original json file, preprocess it and place it into a database for easier use.

## 2. Create dataset
Read the databaseses of comments and response pairs into memory and place them within a dataset 
for further processing. It ultimately returns a list of tensors for the parent comments, a list of tensors for the reply
comments and the tokenizers for them.

## 3. Training
It is composed of two methods, the train_step and the loss_function. The train_step has as
parameters the encoder along with its inputs and its hidden states, the decoder, the target corpus
tokenize, the size of the batch of data being transmitted and the optimizer used for training.


It is composed of two methods, the train_step and the loss_function. The train_step has as
parameters the encoder along with its inputs and its hidden states, the decoder, the target corpus
tokenize, the size of the batch of data being transmitted and the optimizer used for training.


It first initializes the loss to zero and initializes a gradient tape, this is an object provided by
TensorFlow in order to record operations for differentiation. Followed by calling the call
function of the encoder with the parameters being the encoders next input and the encoders
hidden state and it returns the encoders output and the encoders hidden state. The functionality of
the encoder class is explained in the encoder.py subchapter of this paper.


Next the attention is moved to the decoding part of the model, fist the decoder hidden state is set
to be equal to the encoders hidden state which was outputted by the encoders call function. The
last hidden state of the encoder is also known as the context vector explained in the theoretical
substantiation chapter. The decoders input will always be the start token of the sentence so we
set it to that through the help of a TensorFlow method for dimensional expansion.


Teacher forcing is then used for the next part of the training, this method has not been explained
in the theoretical substantiation however it is quite simply a technique where the target word or
the word that we wish to be predicted, is forced to be the next input to the decoder, while its
actual output is used to calculate the loss. So in this step we feed the target sentence through the
decoder while using the context vector provided by the encoder, this way the model is trained to
