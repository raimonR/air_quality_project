import pandas as pd
import requests as r
import numpy as np
import time


def nearest_data(coords_o: tuple, locs: dict):
    min_dist = 100000
    id = None
    for j in locs:
        if j.get('coordinates') is None:
            continue
        coords_n = (j['coordinates']['latitude'], j['coordinates']['longitude'])
        dist = distance(coords_o, coords_n)
        if dist <= min_dist:
            min_dist = dist
            id = j['id']

    return id


def distance(c1: tuple, c2: tuple):
    radius = 6371
    c1 = np.deg2rad(c1)
    c2 = np.deg2rad(c2)
    lat_d = c2[0] - c1[0]
    lon_d = c2[1] - c1[1]

    dist = 2*radius*np.arcsin(np.sqrt(np.sin(lat_d/2)**2 + np.cos(c1[0])*np.cos(c2[0])*np.sin(lon_d/2)**2))
    return dist


def to_dataframe(data: list):
    dfs = []
    for k, point in enumerate(data):
        s1 = pd.Series({'id': point['locationId'], 'pm25 value': point['value'], 'date': point['date']['utc']})
        dfs.append(s1)

    output = pd.concat(dfs, join='inner', axis=1).transpose()
    return output


df = pd.read_pickle('dataset/weather_data/postprocessed/all_weather_data.pkl.zip')
df = df.dropna(axis=0, subset=['LATITUDE', "LONGITUDE"])
df = df.assign(id=np.zeros((df.shape[0]), dtype=int))

rows_to_drop = []
loc_list = []
dft_aq = []

iterations = df[['LATITUDE', 'LONGITUDE']].to_numpy()
iterations = np.unique(iterations, axis=0)
for i, row in enumerate(iterations):
    if i in {1166, 3021, 3097, 4619, 4692, 5542, 5829, 5909, 5971, 6012, 6066, 6266, 6280, 6314, 6340, 6437, 6470, 6523,
             6586, 6605, 6652, 6682, 6795, 6879, 10047, 10091}:
        drop_index = df.index[(df['LATITUDE'] == row[0]) & (df['LONGITUDE'] == row[1])].to_list()
        rows_to_drop.extend(drop_index)
        print(i)
        continue

    coords = (row[0], row[1])
    query = {'date_from': '2020-01-01T00:00:00+00:00', 'date_to': '2021-01-01T00:00:00+00:00', 'parameter': 'pm25',
             'coordinates': f'{coords[0]},{coords[1]}', 'radius': 100000}
    response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/locations", params=query)
    if response.status_code != 200:
        time.sleep(5)
        response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/locations", params=query)

    if response:
        response = response.json()['results']
        aq_loc = nearest_data(coords, response)
    else:
        aq_loc = None

    if aq_loc is None:
        drop_index = df.index[(df['LATITUDE'] == row[0]) & (df['LONGITUDE'] == row[1])].to_list()
        rows_to_drop.extend(drop_index)
        print(i)
    else:
        df.loc[(df['LATITUDE'] == row[0]) & (df['LONGITUDE'] == row[1]), 'id'] = aq_loc
        query = {'date_from': '2020-01-01T00:00:00+00:00', 'date_to': '2021-01-01T00:00:00+00:00', 'parameter': 'pm25',
                 'location_id': aq_loc, 'limit': 100000, 'sort': 'asc'}
        response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/measurements", params=query)
        if response.status_code != 200:
            time.sleep(5)
            response = r.get("https://u50g7n0cbj.execute-api.us-east-1.amazonaws.com/v2/measurements", params=query)

        response = response.json()['results']

        if response:
            loc_list.append(aq_loc)
            df_temp = to_dataframe(response)
            dft_aq.append(df_temp)
        else:
            drop_index = df.index[(df['LATITUDE'] == row[0]) & (df['LONGITUDE'] == row[1])].to_list()
            rows_to_drop.extend(drop_index)
        time.sleep(0.05)

dft_aq = pd.concat(dft_aq, axis=0, ignore_index=True)
dft_aq.to_pickle('dataset/weather_data/postprocessed/air_quality_data.pkl.zip')

df_loc = pd.DataFrame(loc_list, columns=['id']).drop_duplicates().reset_index(drop=True)
df_loc.to_pickle('dataset/weather_data/postprocessed/loc_list.pkl')

df_merge = df.drop(index=rows_to_drop).reset_index(drop=True)
df_merge.to_pickle('dataset/weather_data/postprocessed/weather_data_merged.pkl.zip')
