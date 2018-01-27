# coding: utf-8
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from itertools import chain
import re

MAX_WORDS_PER_LINE = 15
MAX_CHAR_PER_WORD = 15
NA = '_'
# remove harakat
harakat_re = r'(ٍ|َ|ُ|ِ|ّ|ْ|ً)'
# remove every non arabic charachter that isnt a whitespace
nonarabic_nonspace_re = r'[^\u0621-\u064A\s]'
# normalize whitespace
whitespace_re = r'[^\S\n]'
# remove word extensions e.g. أهـــــــــلا
extension_re = r'ـ'
# normalize alef
alef_re = r'(آ|أ|إ|آ)'
# normalize waw_hamzah
waw_hamzah_re = r'(ؤ)'
# normalize yaa_hamzah
yaa_hamzah_re = r'(ئ)'
# normalize taa marbootah
taa_marbootah_re = r'(ة)'


def normalize(poem):
    poem_normalized = re.sub(extension_re, '',
                             poem, flags=re.UNICODE)
    poem_normalized = re.sub(harakat_re, '',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(nonarabic_nonspace_re, ' ',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(whitespace_re, ' ',
                             poem_normalized, flags=re.UNICODE)
    # compact repetitive chars
    poem_normalized = re.sub(r'(.)\1{3,}', r'\1',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(alef_re, 'ا',
                             poem_normalized, flags=re.UNICODE)
    # poem_normalized = re.sub(waw_hamzah_re, 'و',
    #                          poem_normalized, flags=re.UNICODE)
    # poem_normalized = re.sub(yaa_hamzah_re, 'ي',
    #                          poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(taa_marbootah_re, 'ه',
                             poem_normalized, flags=re.UNICODE)
    return poem_normalized


def create_w2v_model(poems):
    verses = [verse for poem in poems for verse in poem]
    model = Word2Vec(verses, size=300, window=3, sg=1, workers=8, iter=25, max_vocab_size=1000)
    model.save('w2v/word2vec')


if __name__ == '__main__':
    make_w2v = False
    filename = 'data/poems.csv'
    data = pd.read_csv(filename, encoding='utf-8')
    data = data.drop_duplicates()
    len(data)
    data = data[data.poem_text.notnull()]
    poems = data.poem_text.apply(lambda p: normalize(p))
    poems = poems.apply(lambda p: p.split('\n'))
    poems = poems.apply(lambda p: [v.split() for v in p])
    if(make_w2v):
        create_w2v_model(poems)
    poems = poems.apply(lambda p: [v for v in p if 0< len(v) <= MAX_WORDS_PER_LINE])
    poems = poems.apply(lambda p: [[w for w in v if len(w)< MAX_CHAR_PER_WORD] for v in p])
    verses_l = list(chain.from_iterable(poems.tolist()))
    verses = pd.DataFrame(data=verses_l)
    verses = verses.fillna(NA)
    verses.to_csv('data/verses.csv', encoding='utf-8', index=False)

    word_vec = Word2Vec.load('w2v/word2vec').wv
    i2w = dict(enumerate(word_vec.index2word))
    w2i = dict([(v, k) for k, v in i2w.items()])
    N_VOCAB = len(i2w)
    verses_w2v = verses.applymap(lambda v: w2i.get(v, N_VOCAB))
    verses_w2v.to_csv('data/verses_w2v.csv', encoding='utf-8', index=False)
