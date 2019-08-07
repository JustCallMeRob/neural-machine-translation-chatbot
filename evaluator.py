import tensorflow as tf


def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):

    prediction = ''

    # Add the start and end tokens to the sentence
    sentence = '<start> ' + sentence + ' <end>'

    # Convert sentence to a list object
    sentence_list = [sentence]

    # Embed sentence
    enc_inp = inp_lang.texts_to_sequences(sentence_list)
    print(enc_inp)

    # Pad sentence to max length of input
    enc_inp = tf.keras.preprocessing.sequence.pad_sequences(enc_inp, maxlen=max_length_inp, padding='post')
    print(enc_inp)

    # Convert list to a tensor object
    enc_inp = tf.convert_to_tensor(enc_inp)
    print(enc_inp)

    # Initialize the encoders initial hidden state
    enc_hidden = [tf.zeros((1, 256))]

    # Call the forward pass method for the encoder
    enc_out, enc_hidden = encoder(enc_inp, enc_hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    # Show the dimensionality of the encoder and the decoders parameters
    print("Encoder hidden shape(batch size, units):{}, output shape(batch size, sequence length, units):{}".format(enc_hidden.shape, enc_out.shape))
    print("Decoder input shape(batch size, sequence length):{}, hidden shape(batch size, units):{}".format(dec_input.shape, dec_hidden.shape))

    for t in range(max_length_targ):

        print("Decoder input:", dec_input)

        # Call the forward pass method for the decoder
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # Find the index with the largest value across axis of the predictions tensor
        predicted_id = tf.argmax(predictions[0]).numpy()

        print("Predict ID:", predicted_id)
        print("Prediction:", list(targ_lang.word_index.keys())[list(targ_lang.word_index.values()).index(predicted_id)])

        # If the prediction is the <end> token, then we have reached the end of the prediction cycle
        if list(targ_lang.word_index.keys())[list(targ_lang.word_index.values()).index(predicted_id)] == '<end>':
            return prediction, sentence

        # Add prediction to the current predicted sentence
        prediction += list(targ_lang.word_index.keys())[list(targ_lang.word_index.values()).index(predicted_id)] + ' '

        # The predicted is fed back into the model
        dec_input = tf.expand_dims(tf.convert_to_tensor([predicted_id]), 0)

    return prediction, sentence


def predict(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    prediction, sentence = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    print('Input: {}'.format(sentence))
    print('Predicted response: {}'.format(prediction))