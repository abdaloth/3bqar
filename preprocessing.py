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
# remove repeating phrases
repeating_phrases_re = r'(\W|^)(.+)\s\2'
# remove repeating chars
repeating_chars_re = r'(.)\1{3,}'
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
    poem_normalized = re.sub(repeating_chars_re, r'\1',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(alef_re, 'ا',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(taa_marbootah_re, 'ه',
                             poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(repeating_phrases_re, r'\2',
                             poem_normalized, flags=re.UNICODE)
    return poem_normalized


def main(src_file='data/poems.csv', output_file='data/raw_text.txt', author=''):
    data = pd.read_csv(src_file, encoding='utf-8')
    if(author):
        data = data[data.author == author]
    data = data.drop_duplicates()
    data = data[data.poem_text.notnull()]
    poems = data.poem_text.tolist()
    poems = [p.split('\n') for p in poems]
    poems = [[clean_and_normalize(v) for v in p if 0 < len(v.split()) <= MAX_WORDS_PER_LINE]
             for p in poems]
    poems = [[' '.join([w for w in v.split() if 0 < len(w) < MAX_CHAR_PER_WORD]
                       ) for v in p] for p in poems]
    verses = list(chain.from_iterable(poems))
    with open(output_file, 'w', encoding='utf8') as f:
        raw_text = '\n'.join(verses)
        f.write(raw_text)


if __name__ == '__main__':
    main()
