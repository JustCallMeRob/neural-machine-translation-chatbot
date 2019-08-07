import tensorflow as tf


# tf.keras.Model groups layers into an object with training and inference features,
# In this situation we instantiate our model by subclassing the Model class.
class Encoder(tf.keras.Model):

    # The layers are defined in __init__
    def __init__(self, vocabulary_size, embedding_dimension, encoding_units, batch_size):

        super(Encoder, self).__init__()

        # tf.keras.layers.Embedding turns positive integers (indexes) into dense vectors of fixed size.
        # eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
        # In the case for textual data, each word has to be represented by a unique integer, this is achievable using the tokenizer api
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dimension)

        # Number of encoding units used in the encoder
        self.encoding_units = encoding_units

        # CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks.
        # GRU (gated recurrent unit) is like a long short-term memory (LSTM) but the forget gate and input gate are combined into an "update gate"
        # encoding_units = number of hidden GRUs inside the encoder
        # return_sequence=True => return the last output in the output sequence
        # return_state=True => return the last state in addition to the output
        # recurrent_initializer => initializer for the recurrent_kernel weights matrix, used for the linear transformation of the recurrent state
        self.gru = tf.keras.layers.CuDNNGRU(self.encoding_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')

        # Size of the batch to be encoded
        self.batch_size = batch_size

    # The models forward pass is implemented here
    def call(self, enc_input, hidden_state_input):

        # Pass the input text through the embedding layer
        enc_input = self.embedding(enc_input)

        # Pass the encoders inputs (embedded text, last hidden state) through the gated recurrent unit layer
        enc_output, hidden_state_output = self.gru(enc_input, initial_state=hidden_state_input)

        # Return the encoder output and the new hidden state
        return enc_output, hidden_state_output

    # Initialize hidden states to zero tensors
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.encoding_units))
