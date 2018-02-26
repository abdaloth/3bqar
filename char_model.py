import numpy as np
from utils import to_categorical, sample

from keras import backend as K
from keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import CuDNNLSTM, Bidirectional, Embedding

with open('data/elia_text.txt', encoding='utf-8') as f:
    verses = [v.strip() for v in f.readlines()]
    word_list = ' '.join(verses).split()
# tokenize words into char index list
raw_text = '\n'.join(verses)
chars = sorted(set(raw_text))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for i, c in enumerate(chars)}

tokenized_text = [c2i[c] for c in raw_text]

# length of input sequences (experiment with this number)
seq_length = 100
chars_in = [tokenized_text[i:i + seq_length]
            for i in range(len(raw_text) - seq_length)]
char_out = [tokenized_text[i + seq_length]
            for i in range(len(raw_text) - seq_length)]
n_patterns = len(chars_in)
# prepare inputs and outputs for model

X = np.reshape(chars_in, (len(chars_in), seq_length, 1))
y = to_categorical(char_out, num_classes=len(chars))
#%%


def build_char_model(weights_path='', input_shape=(X.shape[1], X.shape[2]), output_shape=y.shape[1]):
    inps = Input(shape=input_shape)
    x = CuDNNLSTM(256, return_sequences=True)(inps)
    x = Dropout(0.5)(x)
    x = CuDNNLSTM(256, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = CuDNNLSTM(256)(x)
    x = Dropout(0.5)(x)
    x = Dense(output_shape, activation='softmax')(x)
    model = Model(inps, x)
    if(weights_path):
        model.load_weights(weights_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


K.clear_session()
model = build_char_model()
model.summary()
#%%
tb = TensorBoard(log_dir='logs/char_model_v2', histogram_freq=0,
                 write_graph=True, write_images=False)
mc = ModelCheckpoint('weights/char_v2-epoch-{epoch:02d}-loss-{loss:.4f}.hdf5',
                     save_best_only=True, mode='min',
                     monitor='loss', verbose=1)
model.fit(X, y, epochs=1000, batch_size=1024, callbacks=[tb, mc])
#%%
# GENERATE TEXT
for tmp in [float(i / 10) for i in range(1, 20)]:
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
