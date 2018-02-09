# coding: utf-8
import re
import numpy as np
import pandas as pd
from itertools import chain

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


def clean_and_normalize(poem):
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


if __name__ == '__main__':
    filename = 'data/poems.csv'
    data = pd.read_csv(filename, encoding='utf-8')
    data = data.drop_duplicates()
    len(data)
    data = data[data.poem_text.notnull()]
    poems = data.poem_text.apply(lambda p: clean_and_normalize(p))
    poems = poems.tolist()
    poems = [p.split('\n') for p in poems]
    poems = [[v for v in p if 0< len(v.split()) <= MAX_WORDS_PER_LINE] for p in poems]
    poems = [[' '.join([w for w in v.split() if 0<len(w)< MAX_CHAR_PER_WORD]) for v in p] for p in poems]
    verses = list(chain.from_iterable(poems))
    with open('data/raw_text.txt', 'w', encoding='utf8') as f:
        raw_text = '\n'.join(verses)
        f.write(raw_text)
