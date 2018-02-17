import numpy as np
from utils import word_list, verses, to_categorical, sample

from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding


# tokenize words into char index list
raw_text = '\n'.join(verses)
chars = sorted(set(raw_text))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

tokenized_text = np.array([c2i[c] for c in raw_text])
tokenized_words = [[c2i[c] for c in w] for w in word_list]

word_lengths = [len(w) for w in word_list]
verse_lengths = [len(v.split()) for v in verses]

# length of input sequences (experiment with this number)
seq_length = int(np.percentile(verse_lengths, 90) * np.mean(word_lengths))
stride_offset = 0 # and this number

# prepare inputs and outputs for model
idx = np.arange(seq_length, len(raw_text), seq_length + stride_offset)
chars_in = np.split(tokenized_text, idx)
char_out = np.array(tokenized_text)[idx]


if(not len(chars_in[-1]) == seq_length):
    chars_in = chars_in[:-1]

chars_in = np.stack(chars_in)

X = np.reshape(chars_in, (chars_in.shape[0], seq_length, 1))
y = to_categorical(char_out, num_classes=len(chars))
#%%
K.clear_session()
# build model
inps = Input(shape=(X.shape[1], X.shape[2]))
x = Bidirectional(CuDNNLSTM(256, return_sequences=True))(inps)
x = Dropout(0.3)(x)
x = Bidirectional(CuDNNLSTM(256))(x)
x = Dense(y.shape[1], activation='softmax')(x)
model = Model(inps, x)
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
#%%
tb = TensorBoard(log_dir='logs/char_model', histogram_freq=0,
                 write_graph=True, write_images=False)
mc = ModelCheckpoint('weights/char-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5',
                     save_best_only=True, mode='min',
                     monitor='loss', verbose=1)

model.fit(X, y, epochs=600, batch_size=1024, callbacks=[tb, mc])
#%%
# GENERATE TEXT
for tmp in [.1, .3, .5, .7, 1.2]:
    print(f'temperature = {tmp}')
    idx = np.random.randint(0, X.shape[0] - seq_length)
    sequence = chars_in[idx]
    gen_text = ''
    # generate words
    for i in range(4 * seq_length):
        x_in = np.reshape(sequence, (1, X.shape[1], X.shape[2]))
        prediction = model.predict(x_in, verbose=0)[0]
        # returns the index of the predicted word
        index = sample(prediction, tmp)
        gen_text += i2c[index]
        sequence[:-1] = sequence[1:]
        sequence[-1] = index

    print(gen_text)
    print('-' * 42)
