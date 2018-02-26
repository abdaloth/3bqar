import numpy as np
from utils import *

from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding, Lambda, Add

with open('data/elia_text.txt', encoding='utf-8') as f:
    verses = [v.strip() for v in f.readlines()]
    word_list = ' '.join(verses).split()


w2v = get_word_vector('w2v/elia_w2v', verses)
UNKNOWN = '_'  # token for out-of-vocab
N_VOCAB = len(w2v.vocab) + 1
G = 0.4  # gating value
# length of input sequences (experiment with this number)
seq_length = int(np.mean([len(v.split()) for v in verses]))
# maximum length of chars to consider
# I'm using only the last 2 to preserve the rhyme
MAX_CHAR_PER_WORD = 2
# tokenize words into indices
tokenized_words = [word2index(w, w2v) for w in word_list]

# tokenize words into char index list
chars = sorted(set(UNKNOWN.join(verses)))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}
tokenized_chars = [[c2i[c] for c in index2word(w, w2v)] for w in tokenized_words]

# prepare inputs and outputs for model

padded_chars = pad_sequences(tokenized_chars, maxlen=MAX_CHAR_PER_WORD)
X_ch = np.array([padded_chars[i:i+seq_length] for i in range(len(tokenized_chars)-seq_length)])
X_w = np.array([tokenized_words[i:i+seq_length] for i in range(len(tokenized_words)-seq_length)])
y = np.array([tokenized_words[i+seq_length] for i in range(len(tokenized_words)-seq_length)])

# create embedding_matrix
def get_embedding(w2v):
    w2v_matrix = w2v.syn0
    embedding_matrix = np.zeros((w2v_matrix.shape[0] + 1, w2v_matrix.shape[1]))
    embedding_matrix[1:] = w2v_matrix
    embedding_matrix[0] = np.mean(w2v_matrix, axis=0)
    embedding_layer = Embedding(embedding_matrix.shape[0],
                           embedding_matrix.shape[1],
                           weights=[embedding_matrix],
                           input_length=seq_length,
                           trainable=False)
    return embedding_layer

#%%
def build_word_char_model(weights_path='', input_shapes=[(seq_length, X_ch.shape[2]), (seq_length,)], output_shape=N_VOCAB):
    char_input = Input(shape=input_shapes[0], name='chars_in')
    char_embed = Bidirectional(CuDNNLSTM(256, return_sequences=True),
                               merge_mode='sum')(char_input)

    word_input = Input(shape=input_shapes[1], name='words_in')
    word_embed = get_embedding(w2v)(word_input)

    # NOTE consider doing this dynammically ( with a learnable G param)
    mul_1 = Lambda(lambda x: x * G)(char_embed)
    mul_2 = Lambda(lambda x: x * (1 - G))(word_embed)
    merged = Add()([mul_1, mul_2])

    x = CuDNNLSTM(256, return_sequences=True, name='lstm_word_1')(merged)
    x = Dropout(.5)(x)
    x = CuDNNLSTM(256, name='lstm_word_2')(x)
    x = Dropout(.5)(x)
    next_word = Dense(output_shape, activation='softmax')(x)
    model = Model(inputs=[char_input, word_input], outputs=next_word)
    if(weights_path):
        model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

K.clear_session()
model = build_word_char_model()
model.summary()
#%%
tb = TensorBoard(log_dir='logs/word_char_v2', histogram_freq=0,
                 write_graph=True, write_images=False)
mc = ModelCheckpoint('weights/word_char_v2-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5',
                     save_best_only=True, save_weights_only=True, mode='min',
                     monitor='loss', verbose=1)

generator = PoemSequence(inputs=[X_ch, X_w], output=y,
                         batch_size=1024, num_classes=N_VOCAB)
model.fit_generator(generator=generator, epochs=1000, callbacks=[tb, mc])
model.fit_generator(generator=generator, initial_epoch=1000, epochs=2000, callbacks=[tb, mc])
#%%
# GENERATE TEXT
from random import randint
for tmp in [float(i / 10) for i in range(1, 20)]:
    print(f'temperature = {tmp}')
    idx = randint(0, len(X_w))
    w_in = X_w[idx]
    c_in = X_ch[idx]
    sequence = c_in, w_in
    gen_text = ''
    # generate words
    for i in range(80):
        if(i % 8 == 0):
            gen_text += '\n'
        x_c, x_w = sequence
        x_w = np.reshape(x_w, (1, seq_length,))
        x_c = np.reshape(x_c, (1, seq_length, MAX_CHAR_PER_WORD))
        prediction = model.predict([x_c, x_w], verbose=0)[0]
        # returns the index of the predicted word
        index = sample(prediction, tmp)
        gen_text += ' ' + index2word(index, w2v)
        x_w[:-1] = x_w[1:]
        x_w[-1] = index
        char_idx = [c2i[c] for c in index2word(index, w2v)]
        x_c[:-1] = x_c[1:]
        x_c[-1] = pad_sequences([char_idx], maxlen=MAX_CHAR_PER_WORD)
        sequence = x_c, x_w

    print(gen_text)
    print('-' * 42)
