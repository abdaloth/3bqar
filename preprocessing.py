# coding: utf-8
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import itertools
import re

MAX_WORDS_PER_LINE = 15
MAX_CHAR_PER_WORD = 15
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
    model = Word2Vec(verses, size=300, window=3, sg=1, workers=4, iter=25)
    model.save('word2vec')

# %%
if __name__ == '__main__':
    filename = 'dataset/poems.csv'
    data = pd.read_csv(filename, encoding='utf-8')
    data = data.drop_duplicates()
    len(data)
    data = data[data.poem_text.notnull()]
    poems = data.poem_text.apply(lambda p: normalize(p))
    poems = poems.apply(lambda p: p.split('\n'))
    poems = poems.apply(lambda p: [v.split() for v in p])
    # create_w2v_model(poems)
    poems = [p for p in poems if len(p) <= MAX_WORDS_PER_LINE]
    poems = [[word for word in poem  if len(word)< MAX_CHAR_PER_WORD] for poem in poems]
    poems = [' '.join(p) for p in poems]
    raw_text = '\n'.join(poems)
    with open('normalized_poem_text.txt', 'w', encoding='utf8') as f:
        f.write(raw_text)
