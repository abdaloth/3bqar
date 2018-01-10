from translate import Translator

import pandas as pd
import numpy as np

import pycountry
import json
import folium
from branca.utilities import split_six

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

# %%
world_countries = list(pycountry.countries)
# do this avoids the lists being mismatched
sorted_keys = sorted(translated_countries.keys(), key=lambda k:translated_countries[k])
poem_count = sorted(translated_countries.values())

country_ids = [c_id.alpha_3 for k in sorted_keys for c_id in world_countries if k in c_id.name]
country_ids.remove('SSD')
dt = pd.DataFrame({'Country':country_ids, 'Poems':poem_count})

m = folium.Map(location=[30, 40], zoom_start=4, tiles='cartodbpositron')
geo_path = 'mea.geo.json'
m.choropleth(
 geo_data=geo_path,
 name='choropleth',
 data=dt,
 columns=['Country', 'Poems'],
 key_on='feature.properties.iso_a3',
 fill_color='PuBuGn',
 fill_opacity=0.9,
 line_opacity=0.2,
 legend_name='poems per country',
 threshold_scale=[50,200,500,1500,4200,5000] ,
 reset=True)


with open(geo_path) as f:
    j = json.load(f)
geojson = [{'type': j['type'], 'features': [f]} for f in j['features']]

for gj in map(lambda gj: folium.GeoJson(gj), geojson):
    c_id = gj.data['features'][0]['properties']['iso_a3']
    p = str(dt.loc[dt.Country == c_id, 'Poems'].values) + ' قصيدة'
    gj.add_child(folium.Popup(p))
    gj.line_opacity = 0.3
    gj.add_to(m)

# folium.LayerControl().add_to(m)
# Save to html
m.save('new.html')
