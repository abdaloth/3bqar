import numpy as np
from utils import *

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding, Lambda, Add


w2v = get_word_vector()

UNKNOWN = '_'  # token for out-of-vocab
N_VOCAB = len(w2v.vocab) + 1
G = 0.4  # gating value
seq_length = 1  # play with this number
# maximum length of chars to consider
# I'm using only the last 2 to preserve the rhyme
MAX_CHAR_PER_WORD = 2
# tokenize words into indices
tokenized_words = [word2index(w, w2v) for w in word_list]

# tokenize words into char index list
chars = sorted(set(UNKNOWN.join(verses)))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
tokenized_chars = [[c2i[c]
                    for c in index2word(w, w2v)] for w in tokenized_words]


# prepare inputs and outputs for model
X_w = np.array(tokenized_words)
y = X_w[1:]
X_w = np.reshape(X_w[:-1], (X_w.shape[0] - 1, seq_length))

X_ch = pad_sequences(tokenized_chars, maxlen=MAX_CHAR_PER_WORD)
X_ch = np.reshape(X_ch[:-1], (X_ch.shape[0] - 1, seq_length, X_ch.shape[1]))
#%%
# create embedding_matrix
w2v_matrix = w2v.syn0
embedding_matrix = np.zeros((w2v_matrix.shape[0] + 1, w2v_matrix.shape[1]))
embedding_matrix[1:] = w2v_matrix
embedding_matrix[0] = np.mean(w2v_matrix, axis=0)

#%%
K.clear_session()
# build model
char_input = Input(shape=(seq_length, X_ch.shape[2]), name='chars_in')
char_embed = Bidirectional(CuDNNLSTM(256, return_sequences=True),
                           merge_mode='sum')(char_input)

word_input = Input(shape=(seq_length,), name='words_in')
word_embed = Embedding(embedding_matrix.shape[0],
                       embedding_matrix.shape[1],
                       weights=[embedding_matrix],
                       input_length=seq_length,
                       trainable=False)(word_input)

# NOTE consider doing this dynammically ( with a learnable G param)
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
                         batch_size=1024, num_classes=N_VOCAB)
model.fit_generator(generator=generator, epochs=1, callbacks=[tb, mc])

#%%
# GENERATE TEXT
from random import randint
for tmp in [.3, .5, 1., 1.2, 1.7]:
    print(f'temperature = {tmp}')
    idx = randint(0, len(X_w))
    w_in = X_w[idx:idx + 1]
    c_in = X_ch[idx:idx + 1]
    sequence = c_in, w_in
    gen_text = ''
    # generate words
    # NOTE consider adding temperature
    for i in range(80):
        if(i % 8 == 0):
            gen_text += '\n'
        x_c, x_w = sequence
        prediction = model.predict([x_c, x_w], verbose=0)[0]
        # returns the index of the predicted word
        index = sample(prediction, tmp)
        gen_text += ' ' + word2index(index)
        x_w[0][0] = index
        char_idx = [c2i[c] for c in index2word(index)]
        x_c[0] = pad_sequences([char_idx], maxlen=MAX_CHAR_PER_WORD)
        sequence = x_c, x_w

    print(gen_text)
    print('-' * 42)
