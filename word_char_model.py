import numpy as np
import pandas as pd
from itertools import chain
from preprocessing import MAX_CHAR_PER_WORD, Tokenizer, pad_sequences

from keras import backend as K
from keras.utils import Sequence, to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding, Lambda, Add


class PoemSequence(Sequence):

    def __init__(self, inputs, output, batch_size, num_classes):
        self.x_c, self.x_w = inputs
        self.y = output
        self.batch_size = batch_size
        self.num_classes = num_classes

    def __len__(self):
        return round(len(self.y) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = [self.x_c[(idx * self.batch_size):((idx + 1) * self.batch_size)
                            ], self.x_w[(idx * self.batch_size):((idx + 1) * self.batch_size)]]
        batch_y = self.y[(idx * self.batch_size):((idx + 1) * self.batch_size)]

        return batch_x, to_categorical(batch_y, num_classes=self.num_classes)


with open('data/raw_text.txt', encoding='utf-8') as f:
    verse_list = [v.strip() for v in f.readlines()]
    raw_text = ' '.join(verse_list)
    word_list = raw_text.split()

UNKNOWN = '_'
G = 0.2
seq_length = 1  # play with this number
step_size = 1  # and this number

word_tokenizer = Tokenizer(oov_token=UNKNOWN)
word_tokenizer.fit_on_texts(verse_list)
tokenized_words = word_tokenizer.texts_to_sequences(word_list)

char_tokenizer = Tokenizer(char_level=True, oov_token=UNKNOWN)
char_tokenizer.fit_on_texts(verse_list)
tokenized_chars = char_tokenizer.texts_to_sequences(word_list)

X_w = np.array(list(chain.from_iterable(tokenized_words)))
X_ch = pad_sequences(tokenized_chars, maxlen=MAX_CHAR_PER_WORD)

N_VOCAB = len(word_tokenizer.word_index)
X_ch.shape
y = X_w[1:]
X_w = np.reshape(X_w[:-1], (X_w.shape[0] - 1, seq_length))
X_ch = np.reshape(X_ch[:-1], (X_ch.shape[0] - 1, seq_length, X_ch.shape[1]))
#%%
K.clear_session()

char_input = Input(shape=(seq_length, X_ch.shape[2]), name='chars_in')
char_embed = Bidirectional(CuDNNLSTM(256, return_sequences=True), merge_mode='sum')(char_input)

word_input = Input(shape=(seq_length,), name='words_in')
word_embed = Embedding(input_dim=N_VOCAB, output_dim=256)(word_input)

mul_1 = Lambda(lambda x: x * G)(char_embed)
mul_2 = Lambda(lambda x: x * (1 - G))(word_embed)
merged = Add()([mul_1, mul_2])

x = CuDNNLSTM(256, return_sequences=True, name='lstm_word_1')(merged)
x = Dropout(.5)(x)
x = CuDNNLSTM(256, name='lstm_word_2')(x)
x = Dropout(.5)(x)
next_word = Dense(N_VOCAB, activation='softmax')(x)
model = Model(inputs=[char_input, word_input], outputs=next_word)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
#%%
tb = TensorBoard(log_dir='logs/word_char'.format(model.name), histogram_freq=0,
                     write_graph=True, write_images=False)
mc = ModelCheckpoint('weights/word_char-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5',
                             save_best_only=True, save_weights_only=True, mode='min',
                             monitor='loss', verbose=1)

generator = PoemSequence(inputs=[X_ch, X_w], output=y,
                         batch_size=256, num_classes=N_VOCAB)
model.fit_generator(generator=generator, epochs=1)
#%%
# GENERATE TEXT
from random import randint
idx = randint(0, len(words_in))
w_in = words_in[idx:idx + 1]
w_in.shape
sequence = w_in
gen_text = ''
# generate words
for i in range(30):
    x_w = sequence
    x_c = [pad_w([c2i.get(w, c2i[UNKNOWN]) for c in i2w.get(i, '_')])
           for w in x_w[0]]
    x_c = np.array(x_c).reshape((1, 8, 15))
    x_w = np.array(x_w)
    prediction = model.predict([x_c, x_w], verbose=0)
    # returns the index of the predicted word
    index = np.argmax(prediction[0][:-1])
    gen_text += ' ' + i2w.get(index, '_')
    sequence[0][:-1] = sequence[0][1:]
    sequence[0][-1] = index

print(gen_text)
print('Done.')
