import numpy as np
from gensim.models import Word2Vec
from keras.utils import Sequence, to_categorical


with open('data/raw_text.txt', encoding='utf-8') as f:
    verses = [v.strip() for v in f.readlines()]
    word_list = ' '.join(verses).split()


class PoemSequence(Sequence):
    """turns the poem inputs into a per-batch sequence generator"""

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


def word2index(word, w2v):
    return w2v.vocab[word].index + 1 if word in w2v.vocab else 0


def index2word(index, w2v):
    return w2v.index2word[index - 1]


def get_word_vector():
    # load or create and save word2vec model
    try:
        w2v = Word2Vec.load('w2v/w2v_model').wv
    except FileNotFoundError:
        verse_list = [v.split() for v in verses]
        w2v = Word2Vec(verse_list, window=3, size=256,
                       sg=1, iter=30, workers=8)
        w2v.save('w2v/w2v_model')
        w2v = Word2Vec.load('w2v/w2v_model').wv
    return w2v

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # from keras doc
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)
