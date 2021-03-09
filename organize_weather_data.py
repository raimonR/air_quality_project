import pandas as pd
import numpy as np
import os

file_names = os.listdir('dataset/weather_data/preprocessed/')

dfs = []
columns_to_drop = ['STATION', 'TEMP_ATTRIBUTES', 'DEWP_ATTRIBUTES', 'SLP_ATTRIBUTES', 'STP_ATTRIBUTES',
                   'VISIB_ATTRIBUTES', 'WDSP_ATTRIBUTES', 'MAX_ATTRIBUTES', 'MIN_ATTRIBUTES', 'PRCP_ATTRIBUTES']
column_rename = {'DEWP': 'DEWPOINT', 'SLP': 'SEA LEVEL PRESSURE', 'STP': 'LOCAL PRESSURE', 'VISIB': 'VISIBILITY',
                 'WDSP': 'WINDSPEED', 'MXSPD': 'MAX WINDSPEED', 'MAX': 'MAX TEMP', 'MIN': 'MIN TEMP',
                 'PRCP': 'PRECIPITATION', 'FRSHTT': 'WEATHER'}
for f in file_names:
    temp = pd.read_csv('dataset/weather_data/preprocessed/' + f, parse_dates=[1])
    temp = temp.drop(columns=columns_to_drop)
    temp = temp.rename(columns=column_rename)
    dfs.append(temp)

dft = pd.concat(dfs, axis=0, ignore_index=True).sort_values(by=['DATE', 'NAME']).reset_index(drop=True)

na_1 = dict.fromkeys(['TEMP', 'DEWPOINT', 'SEA LEVEL PRESSURE', 'LOCAL PRESSURE', 'MAX TEMP', 'MIN TEMP'], 9999.9)
na_2 = dict.fromkeys(['VISIBILITY', 'WINDSPEED', 'MAX WINDSPEED', 'GUST', 'SNDP'], 999.9)
na_3 = dict.fromkeys(['PRECIPITATION'], 99.99)
dft = dft.replace(na_1, np.nan)
dft = dft.replace(na_2, np.nan)
dft = dft.replace(na_3, np.nan)
dft = dft.dropna(axis=0, subset=['LATITUDE', "LONGITUDE"])

dft.to_pickle('dataset/weather_data/postprocessed/all_weather_data.pkl.zip')
