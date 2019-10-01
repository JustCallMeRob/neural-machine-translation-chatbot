import pandas as pd
import sqlite3
import tensorflow as tf
from sklearn.model_selection import train_test_split
from encoder import Encoder
from decoder import Decoder
import pickle
import numpy as np
import time  
from evaluator import predict
import os

# Create the dataset from the given timeframe
def create_dataset(timeframes):

    for timeframe in timeframes:
        connection = sqlite3.connect('databases/{}.db'.format(timeframe))
        cursor = connection.cursor()
        limit = 100
        last_unix = 0
        curent_length = limit
        counter = 0
        test_done = False
        parent_list = []
        comment_list = []

        while curent_length == limit:
            dataframe = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {}"
                                    " AND parent NOT NULL ORDER BY unix ASC LIMIT {}".format(last_unix, limit),
                                    connection)
            last_unix = dataframe.tail(1)['unix'].values[0]
            curent_length = len(dataframe)
            if not test_done:
                with open("datasets/test.parent", 'a', encoding='utf8') as f:
                    for content in dataframe['parent'].values:
                        f.write(content + '\n')
                with open("datasets/test.comment", 'a', encoding='utf8') as f:
                    for content in dataframe['comment'].values:
                        f.write(content + '\n')
                test_done = True

            else:
                with open("datasets/train.parent", 'a', encoding='utf8') as f:
                    for content in dataframe['parent'].values:
                        f.write(content + '\n')
                        parent_list.append(content)
                with open("datasets/train.comment", 'a', encoding='utf8') as f:
                    for content in dataframe['comment'].values:
                        f.write(content + '\n')
                        comment_list.append(content)

            counter += 1
            if counter % 20 == 0:
                print(counter * limit, 'rows completed')

        pairs = zip(parent_list, comment_list)
        pair_list = list(pairs)
    return zip(*pair_list)


# Create the tensors and the tokenizers for the provided list of senteces
def tokenize(sentence_list):
    # Initialize the word tokenizer
    # Keep only the 20000 most common words, replace everything else with <unknown>
    # Convert all leters to lowercase characters
    # Filter out some symbols
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000,
                                                      lower=True,
                                                      filters='#$%&/@[\\]^_`{|}~\t\n',
                                                      oov_token="<unknown>")
    # Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency
    tokenizer.fit_on_texts(sentence_list)
    # Transforms each text to a sequence of integers
    tensor = tokenizer.texts_to_sequences(sentence_list)
    # Pad out the matrix so it the size matches up with the longest tensor
    ml = max(len(t) for t in tensor)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, maxlen=ml, padding='post')
    # Show an example
    show_index_to_word_mapping(tokenizer, tensor[0])
    return tensor, tokenizer

# Read and load the dataset into memory
def load_dataset(timeframes):
    parents, comments = create_dataset(timeframes)
    print(parents[:3])
    print(type(parents))
    print(comments[:3])
    print(type(comments))
    parent_tensor, parent_tokenizer = tokenize(parents)
    comment_tensor, comment_tokenizer = tokenize(comments)
    return parent_tensor, comment_tensor, parent_tokenizer, comment_tokenizer


# Show word to index mapping for provided tensor
def show_index_to_word_mapping(tokenizer, tensor):
    print("Original tensor:")
    print(tensor)
    print("Mapping:")
    for t in tensor:
        if t != 0:
            print("%s = %d" % (list(tokenizer.word_index.keys())[list(tokenizer.word_index.values()).index(t)], t))
