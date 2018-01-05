# coding: utf-8
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import itertools
import re

# remove harakat
harakat_re = r'(ٍ|َ|ُ|ِ|ّ|ْ|ً)'
# remove every non arabic charachter that isnt a space or a newline
nonarabic_nonspace_re = r'[^\u0621-\u064A \n]'
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
    # compact repetitive spaces
    poem_normalized = re.sub(r'( ){2,}', r' ',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(alef_re, 'ا',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(waw_hamzah_re, 'و',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(yaa_hamzah_re, 'ي',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(taa_marbootah_re, 'ه',
                             poem_normalized, flags=re.UNICODE)
    return poem_normalized


def create_w2v_model(verses):
    verses = [s.split() for v in verses for s in v]
    model = Word2Vec(verses,size=300, window=3,sg=1,workers=4,iter=25)
    model.save('word2vec')

# %%
if __name__ == '__main__':
    filename = 'dataset/poems.csv'
    data = pd.read_csv(filename, encoding='utf-8')
    data = data.drop_duplicates()
    len(data)
    data = data[data.poem_text.notnull()]
    poems = data.poem_text.apply(lambda p: normalize(p).strip())
    poems = poems.drop_duplicates()
    len(poems)
    # poems = poems.apply(lambda x: x.split('\n'))
    poems = poems.tolist()
    verses = [p.split('\n') for p in poems]
    # create_w2v_model(verses)
    poems = list(itertools.chain.from_iterable(verses))
    strip = str.rstrip
    raw_text = '\n'.join([line for line in poems if(1 < len(strip(line, '\n')) <=100)]).rstrip('\n')
    # raw_text = re.sub(r'\s{2,}', '\n', raw_text, flags=re.UNICODE)
    raw_text[:100]
    with open('normalized_poem_text.txt', 'w', encoding='utf8') as f:
        f.write(raw_text)
