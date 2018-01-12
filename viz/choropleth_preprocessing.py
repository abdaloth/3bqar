import pycountry
import numpy as np
import pandas as pd
from translate import Translator

df = pd.read_csv('../dataset/poems.csv', encoding='utf8')
# drop poems with no country
data = df.drop(df[df.section.str.contains(' ')].index)
data = data.dropna(subset=['section'])
countries = data.section.value_counts()
countries = dict(countries)
del countries['section']
# translate from arabic to get the iso_a3 code
translator = Translator(provider='mymemory', from_lang='ar',to_lang='en')
translated_countries = {}
for k in sorted(countries.keys()):
    translated = translator.translate(k).lower().strip(' .')
    translated_countries[translated.title()] = countries[k]

world_countries = list(pycountry.countries)
# do this avoids the lists being mismatched
sorted_keys = sorted(translated_countries.keys(), key=lambda k:translated_countries[k])
poem_count = sorted(translated_countries.values())

country_ids = [c_id.alpha_3 for k in sorted_keys for c_id in world_countries if k in c_id.name]
country_ids.remove('SSD')
dt = pd.DataFrame({'Country':country_ids, 'Poems':poem_count})

dt.to_csv('poems_per_country.csv')
