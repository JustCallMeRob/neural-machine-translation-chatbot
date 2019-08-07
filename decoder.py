import tensorflow as tf


# tf.keras.Model groups layers into an object with training and inference features,
# In this situation we instantiate our model by subclassing the Model class.
class Decoder(tf.keras.Model):

    # The layers are defined in __init__
    def __init__(self, vocabulary_size, embedding_dim, decoding_units, batch_size):

        super(Decoder, self).__init__()

        # Size of the batch to be decoded
        self.batch_size = batch_size

        # Number of decoding units inside the decoder
        self.dec_units = decoding_units

        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_dim)

        # Gated recurrent unit layer
        self.gru = tf.keras.layers.CuDNNGRU(self.dec_units,
                                            return_sequences=True,
                                            return_state=True,
                                            recurrent_initializer='glorot_uniform')

        # Regular densely connected layer for calculating decoder output
        # output = activation(dot(input, kernel) + bias)
        self.fc = tf.keras.layers.Dense(vocabulary_size)

        # Regular densely connected layers used for calculating Bahdanau attention
        self.FC1 = tf.keras.layers.Dense(self.dec_units)
        self.FC2 = tf.keras.layers.Dense(self.dec_units)
        self.FC = tf.keras.layers.Dense(1)

    # The models forward pass is implemented here
    def call(self, dec_input, hidden_state_input, enc_output):

        # enc_output shape:(batch_size, max_length, encoding_units)
        # hidden_state_input shape:(batch_size, decoding_units)
        # hidden_with_time_axis shape:(batch_size, 1, decoding_units)
        # we are doing this to be able to perform addition with encoder_output for score calculation
        hidden_with_time_axis = tf.expand_dims(hidden_state_input, 1)

        # score = FC(tanh(FC(EO) + FC(H)))
        # score shape:(batch_size, max_length, 1)
        score = self.FC(tf.nn.tanh(self.FC1(enc_output) + self.FC2(hidden_with_time_axis)))

        # the attention_weights are simply a softmax of the score tensor
        # attention_weights shape:(batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context vector = sum(attention_weights * encoder_output, axis = 1)
        # context_vector shape:(batch_size, hidden_size)
        context_vector = tf.reduce_sum(attention_weights * enc_output, axis=1)

        # dec_input shape after embedding:(batch_size, 1, embedding_dim)
        dec_input = self.embedding(dec_input)

        # dec_input shape after concatenation:(batch_size, 1, embedding_dim + hidden_size)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)

        # passing the concatenated vector to the GRU
        dec_output, hidden_state_output = self.gru(dec_input)

        # output shape:(batch_size * 1, hidden_size)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pass the dec_out through a densely connected layer
        # output shape:(batch_size * 1, vocab)
        dec_output = self.fc(dec_output)

        return dec_output, hidden_state_output, attention_weights

    # Initialize hidden states to zero tensors
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.dec_units))