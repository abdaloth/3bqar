import pandas as pd
import json
import folium
from branca.utilities import split_six

dt = pd.read_csv('poems_per_country.csv')

geo_path = 'mea.geo.json'
m = folium.Map(location=[30, 40], zoom_start=4, tiles='stamenwatercolor')
m.choropleth(
 geo_data=geo_path,
 name='choropleth',
 data=dt,
 columns=['Country', 'Poems'],
 key_on='feature.properties.iso_a3',
 fill_color='YlGn',
 fill_opacity=0.9,
 line_weight=2,
 line_color='black',
 line_opacity=0.2,
 legend_name='poems per country',
 threshold_scale=[50,200,500,1500,4200,5000] ,
 reset=True)


with open(geo_path) as f:
    j = json.load(f)

geojson = [{'type': j['type'], 'features': [f]} for f in j['features']]
style_function = lambda x: {'color':'black', 'opacity':.2}

for gj in map(lambda gj: folium.GeoJson(gj, style_function=style_function), geojson):
    c_id = gj.data['features'][0]['properties']['iso_a3']
    p = str(dt.loc[dt.Country == c_id, 'Poems'].values) + ' قصيدة'
    gj.add_child(folium.Popup(p))
    gj.line_opacity = 0.3
    gj.add_to(m)

m.save('new.html')
