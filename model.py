
# coding: utf-8

import numpy as np
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils

poems = np.load('normalized_poem_text.npy')
raw_text = '\n'.join(poems)
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Number of characters in the datatset: ", n_chars)
print("unique charachter count: ", n_vocab)

seq_length = 42  # play with this number
step_size = 1  # and this number
input_sequences = []  # list of sentences of length seq_length
output_chars = []  # list of charachters following that sentence
for i in range(0, n_chars - seq_length, step_size):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    input_sequences.append([char_to_int[char] for char in seq_in])
    output_chars.append(char_to_int[seq_out])

n_seq = len(input_sequences)
print("number of sequences: ", n_seq)

input_sequences = np.array(input_sequences)
output_chars = np.array(output_chars)

# reshape X to be [samples, time steps, features]
# then normalize it
X = np.reshape(input_sequences, (n_seq, seq_length, 1))

# one-hot encode the output
y = np_utils.to_categorical(output_chars, num_classes=n_vocab)

# %%
# save so you don't have to go through that again
np.save('input_sequences', X)
np.save('output_chars', y)
# %% # checkpoint
# load files
X = np.load('input_sequences.npy')
y = np.load('output_chars.npy')
# %%

# build the LSTM model


def build_model(weights=''):
    model = Sequential()
    model.add(LSTM(512, input_shape=(
        X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(512))
    model.add(Dropout(0.1))
    model.add(Dense(y.shape[1], activation='softmax'))

    if weights:
        model.load_weights(weights)
    optimizer = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    return model


model = build_model()
print(model.summary())
# define the checkpoint
w_path = "weights/nizar_v1-weights-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5"
tBoard = TensorBoard(log_dir='./logs/nizar_v0', histogram_freq=0,
                     write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(w_path, monitor='loss', verbose=1,
                             save_best_only=True, save_weights_only=True, mode='min')

# %%
model.fit(X, y, epochs=50, batch_size=1024, callbacks=[checkpoint, tBoard])
del X
del y


# %% ## Generate Text

# chars = # TODO: save char list externally and load it here
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

filename = ''

# model = build_model('weights/weights-epoch-04-loss-1.7161.hdf5')

from random import randint
random_seed = randint(0, len(raw_text) - 42)
seed = raw_text[random_seed:random_seed + 42]
sequence = [char_to_int[char] for char in seed]
gen_text = ''
print('Seed:\n')
print(seed)
# generate characters
for i in range(1000):
    x = np.reshape(sequence, (1, len(sequence), 1))
    x = x / len(chars)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)  # returns the index of the predicted char
    gen_text += int_to_char[index]
    sequence.append(index)
    sequence = sequence[1:len(sequence)]

print(gen_text)
print('Done.')
