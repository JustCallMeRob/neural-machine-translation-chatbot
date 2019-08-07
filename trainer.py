import numpy as np
import tensorflow as tf


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss)


def train_step(encoder, enc_inp, target, target_tokenizer, enc_hidden, decoder, optimizer, batch_size):
    loss = 0

    # Gradient tape is used to record operations for differentiation
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(enc_inp, enc_hidden)

        # Set the decoders hidden state input equal to the encoders hidden state output
        # The last hidden state output of the encoder is also knows as the context vector
        dec_hidden = enc_hidden

        # Set the decoders input to be the start token for a sentence
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * batch_size, 1)

        # Teacher forcing step
        # This is a method for quickly and efficiently train RNN models by feeding the ground truth from a prior time
        # step as input to the next, in this situation we pass the "target word" as the next input to the decoder
        for t in range(1, target.shape[1]):

            dec_output, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            # Get the next loss value and add it to the total loss
            loss += loss_function(target[:, t], dec_output)

            # Use teacher forcing
            dec_input = tf.expand_dims(target[:, t], 1)

    batch_loss = (loss / int(target.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    # Compute the derivative of the loss function relative to the provided variable
    gradients = tape.gradient(loss, variables)

    # Apply computed derivative to the variables
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss
