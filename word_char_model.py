import pandas as pd
import numpy as np
from preprocessing import MAX_CHAR_PER_WORD
from gensim.models import Word2Vec

from keras import backend as K
from keras.utils import to_categorical, Sequence
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding, GlobalMaxPooling1D, concatenate


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


def pad_w(w):
    return w + [c2i[UNKNOWN]] * (MAX_CHAR_PER_WORD - len(w))


def get_embedding_matrix():
    n, d = word_vec.syn0.shape
    embedding_matrix = np.zeros((n + 1, d))
    embedding_matrix[:-1] = word_vec.syn0
    return embedding_matrix


UNKNOWN = '_'
# load word vector and create index2value and value2index dicts
word_vec = Word2Vec.load('w2v/word2vec').wv
i2w = dict(enumerate(word_vec.index2word))
w2i = dict([(v, k) for k, v in i2w.items()])
CLASS_NUM = len(i2w) + 1
# add UNKNOWN to the dictionary for unknown chars
i2c = dict(enumerate(sorted(set(' '.join(word_vec.index2word) + UNKNOWN))))
c2i = dict([(v, k) for k, v in i2c.items()])

# load verses and flatten into one continuous array
X_w = pd.read_csv('data/verses_w2v.csv', encoding='utf-8').as_matrix()
X_w_flat = X_w.flatten()

# load word-chars and flatten into one continuous array
X_ch = pd.read_csv('data/verses.csv', encoding='utf-8')
X_ch = X_ch.applymap(lambda v: pad_w([c2i.get(w, c2i[UNKNOWN]) for w in v]))
X_ch = np.array(X_ch.values.tolist())
X_ch_flat = X_ch.reshape((X_ch.shape[0] * X_ch.shape[1], X_ch.shape[2]))

seq_length = 8  # play with this number
step_size = 3  # and this number

words_in = []
chars_in = []
words_out = []
for i in range(0, X_w_flat.shape[0] - seq_length, step_size):
    words_in.append(X_w_flat[i:i + seq_length])
    chars_in.append(X_ch_flat[i:i + seq_length])
    words_out.append(X_w_flat[i + seq_length])

words_in = np.stack(words_in)
chars_in = np.stack(chars_in)


#%%
K.clear_session()

char_input = Input(shape=(seq_length, MAX_CHAR_PER_WORD,), name='chars_in')
char_embed = Bidirectional(CuDNNLSTM(256, return_sequences=True), merge_mode='sum')(char_input)

word_input = Input(shape=(seq_length,), name='words_in')
embed = get_embedding_matrix()
word_embed = Embedding(output_dim=256, input_dim=embed.shape[0],
                       weights=[embed], trainable=False)(word_input)
concat = concatenate([char_embed, word_embed], axis=-1)
x = CuDNNLSTM(256, return_sequences=True, name='lstm_word_1')(concat)
x = Dropout(.5)(x)
x = CuDNNLSTM(256, name='lstm_word_2')(x)
x = Dropout(.5)(x)
next_word = Dense(CLASS_NUM, activation='softmax')(x)
model = Model(inputs=[char_input, word_input], outputs=next_word)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
#%%
tb = TensorBoard(log_dir='logs/word_char'.format(model.name), histogram_freq=0,
                     write_graph=True, write_images=False)
mc = ModelCheckpoint('weights/word_char0-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5',
                             save_best_only=True, save_weights_only=True, mode='min',
                             monitor='loss', verbose=1)

generator = PoemSequence(inputs=[chars_in, words_in], output=words_out,
                         batch_size=2048, num_classes=CLASS_NUM)
model.fit_generator(generator=generator, callbacks=[tb, mc], epochs=25)
#%%
##GENERATE TEXT
from random import randint
idx = randint(0, len(words_in))
w_in = words_in[idx:idx+1]
w_in.shape
sequence = w_in
gen_text = ''
# generate words
for i in range(30):
    x_w = sequence
    x_c = [pad_w([c2i.get(w, c2i[UNKNOWN]) for c in i2w.get(i, '_')]) for w in x_w[0]]
    x_c = np.array(x_c).reshape((1, 8, 15))
    x_w = np.array(x_w)
    prediction = model.predict([x_c, x_w], verbose=0)
    index = np.argmax(prediction[0][:-1])  # returns the index of the predicted word
    gen_text += ' ' + i2w.get(index, '_')
    sequence[0][:-1] = sequence[0][1:]
    sequence[0][-1] = index

print(gen_text)
print('Done.')
