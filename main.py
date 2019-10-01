import tensorflow as tf   
import pickle
from create_dataset import load_dataset
from encoder import Encoder
from decoder import Decoder
from evaluator import predict
import os
import time
from trainer import train_step
from sklearn.model_selection import train_test_split


# Print dimensionality of the encoders/decoders:inputs, outputs and hidden units
def print_encoder_decoder_dimensions():
    # Initialize example data
    example_input_batch, example_target_batch = next(iter(dataset))
    print(example_input_batch.shape, example_target_batch.shape)

    # Initialize example encoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
    enc_sample_hidden = encoder.initialize_hidden_state()
    enc_sample_output, enc_sample_hidden = encoder.call(example_input_batch, enc_sample_hidden)
    print("Encoder hidden shape(batch size, units):{}, output shape(batch size, sequence length, units):{}".format(
        enc_sample_hidden.shape, enc_sample_output.shape))

    # Initialize example decoder
    decoder = Decoder(vocab_tar_size, embedding_dim, units, buffer_size)
    dec_sample_hidden = enc_sample_hidden
    dec_sample_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * batch_size, 1)
    sample_prediction, dec_sample_hidden, _ = decoder.call(dec_sample_input, dec_sample_hidden, enc_sample_output)
    print("Decoder input shape(batch size, sequence length):{}, hidden shape(batch size, units):{}".format(
        dec_sample_input.shape, dec_sample_hidden.shape))


# Bring everything together
if __name__ == '__main__':

    # Current used databases
    time_frame = ['2015-03']
    # Create dataset flag, if true create dataset files for the current used database
    create = False
    # Train flag, if true train a new model, else use existing trained model
    train = False

    if create:
        # Load the dataset
        input_tensor, target_tensor, input_tokenizer, target_tokenizer = load_dataset(time_frame)

        # Pickle them so as to not require loading from the database again
        pickle.dump(input_tensor, open("tensors/input_tensor.p", "wb"))
        pickle.dump(target_tensor, open("tensors/target_tensor.p", "wb"))
        pickle.dump(input_tokenizer, open("tokenizers/inp_lang.p", "wb"))
        pickle.dump(target_tokenizer, open("tokenizers/targ_lang.p", "wb"))
    else:
        # Load the dataset from pickles
        input_tensor = pickle.load(open("tensors/input_tensor.p", "rb"))
        target_tensor = pickle.load(open("tensors/target_tensor.p", "rb"))
        input_tokenizer = pickle.load(open("tokenizers/inp_lang.p", "rb"))
        target_tokenizer = pickle.load(open("tokenizers/targ_lang.p", "rb"))

    # Creating training and validation sets using an 20% split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2)

    # Show lengths
    print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

    # Number of iteration over the entire dataset
    epochs = 30
    # Size of the input buffer
    buffer_size = len(input_tensor_train)
    # Size of each batch
    batch_size = 10
    # Number of steps for every epoch
    steps_per_epoch = len(input_tensor_train) // batch_size
    # Word embedding is a vectoral representation of the words inputed to the encoder
    # It is a distributed representation where each word is mapped to a fixed sized vector of continuous values
    # embedding_dim represents the size of the vector used to embed each word
    embedding_dim = 256
    # Number of hidden units
    units = 256
    # Size of the input vocabulary
    vocab_inp_size = len(input_tokenizer.word_index) + 1
    # Size of the output vocabulary
    vocab_tar_size = len(target_tokenizer.word_index) + 1

    tf.enable_eager_execution(config=None, device_policy=None, execution_mode=None)

    # Prepare the dataset
    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
    # Combine consecutive elements into batches
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Initialize encoder and decoder
    encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size)

    # Initialize optimizer
    optimizer = tf.train.AdamOptimizer()

    # Checkpointing system for saving progress over every epoch
    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)

    print_encoder_decoder_dimensions()

    # Begin training the model
    if train:
        for epoch in range(epochs):
            start = time.time()
            enc_hidden = encoder.initialize_hidden_state()
            epoch_loss = 0

            # For each parent and comment pairs of batch_size, pass them through the NMT for training
            for (batch, (input_tensor, target_tensor)) in enumerate(dataset.take(steps_per_epoch)):

                batch_loss = train_step(encoder, input_tensor, target_tensor, target_tokenizer, enc_hidden, decoder, optimizer, batch_size)
                epoch_loss += batch_loss

                if batch % 100 == 0:
                    print("Epoch {} Batch {} Loss {:.4f}".format(epoch + 1, batch, batch_loss.numpy()))

            # Saving (checkpoint) every epochs
            checkpoint.save(file_prefix=checkpoint_prefix)

            print("Epoch {} Loss {:.4f}".format(epoch + 1, epoch_loss / steps_per_epoch))
            print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    print("CHECKPOINT: ", tf.train.latest_checkpoint(checkpoint_dir))

    # Get length of tensors
    max_length_inp = max(len(t) for t in input_tensor)
    max_length_targ = max(len(t) for t in target_tensor)

    # Evaluate the trained model with user inputs
    while True:
        inp = input()
        predict(inp, encoder, decoder, input_tokenizer, target_tokenizer, max_length_inp, max_length_targ)
        if inp == "":
            break
