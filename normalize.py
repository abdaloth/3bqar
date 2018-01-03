# coding: utf-8
import pandas as pd
import numpy as np
import re

filename = 'dataset/poems.csv'
data = pd.read_csv(filename, encoding='utf-8')

nizar_poems = data[data.author == 'نزار قباني']
poems = nizar_poems['poem_text']
latin_re = re.compile('[a-zA-Z]') # remove bilingual poems
raw_text = []
for poem in poems:
    try:
        if(re.search(latin_re, poem)):
            continue
        raw_text.append(poem)
    except:
        print(poem)
len(raw_text)

raw_text[1337]

def normalize_poem(poem):
    # remove harakat
    harakat_re = r'(ٍ|َ|ُ|ِ|ّ|ْ|ً)'
    # remove every non arabic charachter that isnt a space, dot or new line
    nonarabic_nonspace_re = r'[^\u0621-\u064A .\n]'
    # remove word extensions e.g. أهـــــــــلا
    extension_re = r'ـ'
    # normalize alef
    alef_re = r'(آ|أ|إ|آ)'
    # normalize waw_hamzah
    waw_hamzah_re = r'(ؤ)'
    # normalize taa marbootah
    taa_marbootah_re = r'(ة)'

    poem_normalized = re.sub(extension_re, '', poem, flags=re.UNICODE)
    poem_normalized = re.sub(harakat_re, '', poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(nonarabic_nonspace_re, ' ', poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(r'( ){2,}', r' ', poem_normalized, flags=re.UNICODE) # compact repetitive spaces
    poem_normalized = re.sub(alef_re, 'ا', poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(waw_hamzah_re, 'و', poem_normalized, flags=re.UNICODE)
    poem_normalized = re.sub(taa_marbootah_re, 'ه', poem_normalized, flags=re.UNICODE)
    return poem_normalized

test = normalize_poem(raw_text[1337])
print(test.split('\n'))

poems_normalized = np.empty(len(raw_text), dtype=object)
for i,poem in enumerate(raw_text):
    poems_normalized[i] = normalize_poem(poem)

np.save('normalized_poem_text', poems_normalized)
