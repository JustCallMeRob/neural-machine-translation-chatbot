# Neural machine translation chatbot 

The purpose of this project is to demonstrate the feasibility and implement methods of creating
artificial conversational entities, or chatterbots using a novel new sequence-to-sequence techniques derived 
from advancements in machine learning, specifically in the fields of natural language processing and 
neural machine translation (NMT).

An NMT will be trained on a dataset of comments from Reddit provided by Jason? i think, from [this link](https://files.pushshift.io/reddit/comments/). Now this repository contains every publicly available comment posted to reddit since 2005, i mostly used three months from 2015. The idea being that if i feed comment and response pairs to the NMT it would map simmilar responses to their comment parents and be able to spit out sensible responses for any input.

# Functional Description
  - Create a dataframe if one is requested and pickle its tensor representation and tokenizers
    for later use, else use existing ones.
    
  - Initialize all of the training parameters such as number of epochs, buffer size, batch size,
    embedding dimension, number of hidden encoding and decoding units and determine the
    vocabulary sizes for the input corpus and the output corpus.
    
  - Create the batches of tensors from the dataset.
  
  - Initialize encoder, decoder and optimizer.
  
  - Initialize a checkpointing system to save the models as they are being trained, in case of
    having to halt the training prematurely or if we want to analyse older models for
    comparisons.
    
  - Train the model if training has been specified, otherwise use an existing trained model.
  
  - Evaluate trained model.
  
 The project is extensibly commented so any additional information can be found in the code itself.

# Neural Network Model
The repository contains a model trained with the following charactersistics: number of epochs = 30, batch size = 10, embedding
dimension = 256, number of hidden recurrent units = 256, optimizer = Adam optimizer, max word length of sentences = 10, final loss = 0.9. Training data consisted of 5000 comment and responses 1-20 words long.
Total train time was one hour. I made better models however the file size was too big for github.

# Results
![results 1](https://github.com/JustCallMeRob/neural-machine-translation-chatbot/blob/master/results/1.PNG)
![results 2](https://github.com/JustCallMeRob/neural-machine-translation-chatbot/blob/master/results/2.PNG)
![results 3](https://github.com/JustCallMeRob/neural-machine-translation-chatbot/blob/master/results/3.PNG)
