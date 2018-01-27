import pandas as pd
import numpy as np
from preprocessing import NA, MAX_CHAR_PER_WORD
from gensim.models import Word2Vec

from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import CuDNNLSTM, Bidirectional, Embedding, concatenate

def pad_w(w):
    return w + [c2i[NA]] * (MAX_CHAR_PER_WORD - len(w))


def get_embedding_matrix():
    n, d = word_vec.syn0.shape
    embedding_matrix = np.zeros((n + 1, d))
    embedding_matrix[:-1] = word_vec.syn0
    return embedding_matrix

# load word vector and create index2value and value2index dicts
word_vec = Word2Vec.load('w2v/word2vec').wv
i2w = dict(enumerate(word_vec.index2word))
w2i = dict([(v, k) for k, v in i2w.items()])
# add NA to the dictionary for unknown chars
i2c = dict(enumerate(sorted(set(' '.join(word_vec.index2word) + NA))))
c2i = dict([(v, k) for k, v in i2c.items()])

# load verses and flatten into one continuous array
X_w = pd.read_csv('data/verses_w2v.csv', encoding='utf-8').as_matrix()
X_w_flat = X_w.flatten()

# load word-chars and flatten into one continuous array
X_ch = pd.read_csv('data/verses.csv', encoding='utf-8')
X_ch = X_ch.applymap(lambda v: pad_w([c2i.get(w, c2i[NA]) for w in v]))
X_ch = np.array(X_ch.values.tolist())
X_ch_flat = X_ch.reshape((X_ch.shape[0]*X_ch.shape[1], X_ch.shape[2]))

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
chars_in =  np.stack(chars_in)
words_out = to_categorical(words_out, num_classes=len(i2w))
chars_in.shape
words_out.shape
X[1]
embed = get_embedding_matrix()
K.clear_session()
char_input = Input(shape=(15, 15,))
word_input = Input(shape=(15,))
word_embed = Embedding(output_dim=300, input_dim=embed.shape[0],
                       weights=[embed], trainable=False)(word_input)
char_embed = Bidirectional(CuDNNLSTM(300, return_sequences=True), merge_mode='sum')(char_input)
concat = concatenate([char_embed, word_embed])
next_word = Dense(y.shape[1], activation='softmax')(concat)
model = Model(inputs=[char_input, word_input], outputs=next_word)
model.compile(optimizer='adam', loss='binary_crossentropy')
model.summary()
# np.zeros((1,15))
