import pandas as pd
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler
rng = default_rng(0)

d = 5
df_w = pd.read_pickle('dataset/weather_data/postprocessed/weather_data_merged.pkl.zip')
df_aq = pd.read_pickle('dataset/weather_data/postprocessed/air_quality_data.pkl.zip')
loc_list = pd.read_pickle('dataset/weather_data/postprocessed/loc_list.pkl')

df_aq['pm25 value'] = pd.to_numeric(df_aq['pm25 value'])
df_aq['date'] = pd.to_datetime(df_aq['date'], utc=True)
df_w['DATE'] = pd.to_datetime(df_w['DATE'], utc=True)

set_x = []
set_y = []

cal = np.arange('2020-01-01 00:00:00.00', '2020-12-26 00:00:00.00', step=np.timedelta64(1, 'D'), dtype='datetime64')
ids = loc_list['id'].sort_values().reset_index(drop=True)
for idx, i in enumerate(ids):
    aq = df_aq.loc[df_aq['id'] == i, ['pm25 value', 'date']]
    aq = pd.DataFrame({'pm25': aq['pm25 value'].values}, index=pd.DatetimeIndex(aq['date']))
    aq = aq[~aq.index.duplicated()]

    dw = df_w.loc[df_w['id'] == i]
    dw = pd.DataFrame(
        {'elevation': dw['ELEVATION'].values, 'temp': dw['TEMP'].values, 'dewpoint': dw['DEWPOINT'].values,
         'p0': dw['SEA LEVEL PRESSURE'].values, 'p': dw['LOCAL PRESSURE'].values, 'visibility': dw['VISIBILITY'].values,
         'windspeed': dw['WINDSPEED'].values, 'max windspeed': dw['MAX WINDSPEED'].values, 'gust': dw['GUST'].values,
         'max temp': dw['MAX TEMP'].values, 'min temp': dw['MIN TEMP'].values, 'rain': dw['PRECIPITATION'].values,
         'snow': dw['SNDP'].values, 'weather': dw['WEATHER'].values}, index=pd.DatetimeIndex(dw['DATE']))
    dw = dw[~dw.index.duplicated()]

    rand_list = rng.choice(a=cal, size=75, replace=False)
    for j in rand_list:
        tx_range = [pd.Timestamp(j, tz='UTC'), pd.Timestamp(j + np.timedelta64(24*d - 1, 'h'), tz='UTC')]
        ty_range = [pd.Timestamp(j + np.timedelta64(24*d, 'h'), tz='UTC'),
                    pd.Timestamp(j + np.timedelta64(24*(d + 1) - 1, 'h'), tz='UTC')]
        aq_x = aq.loc[tx_range[0]:tx_range[1]].copy(deep=True)
        dw_x = dw.loc[tx_range[0]:tx_range[1]].copy(deep=True)

        aq_x.loc[aq_x['pm25'] < 0, 'pm25'] = np.nan
        if (aq_x.isnull().sum()/aq_x.shape[0])[0] < 0.3:
            if (aq_x.shape[0] == 120) and (dw_x.shape[0] == 5):
                dt = aq_x.index.to_series().diff().dt.seconds.div(3600)
                dt = np.ma.array(dt, mask=np.isnan(dt))
                if np.all(dt == 1):
                    aq_x = aq_x.interpolate('time', limit_direction='both')
                    aq_x.loc[aq_x['pm25'] < 0, 'pm25'] = 0

                    # TODO: WHEN FILLING NAN VALUES, PICK BETWEEN MAX OF 0 OR INTERPOLATED VALUE
                    if np.any(dw_x['dewpoint'].isna()):
                        if np.all(dw_x['dewpoint'].isna()):
                            dw_x['dewpoint'] = dw_x['dewpoint'].fillna(50)
                        else:
                            m = dw['dewpoint'].mean()
                            dw_x['dewpoint'] = dw_x['dewpoint'].fillna(m)

                    if np.any(dw_x['p0'].isna()):
                        if np.all(dw_x['p0'].isna()):
                            dw_x['p0'] = dw_x['p0'].fillna(1013.25)
                        else:
                            m = dw['p0'].mean()
                            dw_x['p0'] = dw_x['p0'].fillna(m)

                    if np.any(dw_x['visibility'].isna()):
                        if np.all(dw_x['visibility'].isna()):
                            dw_x['visibility'] = dw_x['visibility'].fillna(5)
                        else:
                            m = dw['visibility'].mean()
                            dw_x['visibility'] = dw_x['visibility'].fillna(m)

                    if np.any(dw_x['windspeed'].isna()):
                        if np.all(dw_x['windspeed'].isna()):
                            dw_x['windspeed'] = dw_x['windspeed'].fillna(0)
                        else:
                            m = dw['windspeed'].mean()
                            dw_x['windspeed'] = dw_x['windspeed'].fillna(m)

                    if np.any(dw_x['max windspeed'].isna()):
                        if np.all(dw_x['max windspeed'].isna()):
                            dw_x['max windspeed'] = dw_x['max windspeed'].fillna(0)
                        else:
                            m = dw['max windspeed'].mean()
                            dw_x['max windspeed'] = dw_x['max windspeed'].fillna(m)

                    if np.any(dw_x['gust'].isna()):
                        if np.all(dw_x['gust'].isna()):
                            dw_x['gust'] = dw_x['gust'].fillna(0)
                        else:
                            m = dw['gust'].mean()
                            dw_x['gust'] = dw_x['gust'].fillna(m)

                    if np.any(dw_x['max temp'].isna()):
                        if np.all(dw_x['max temp'].isna()):
                            dw_x['max temp'] = dw_x['max temp'].fillna(60)
                        else:
                            m = dw['max temp'].mean()
                            dw_x['max temp'] = dw_x['max temp'].fillna(m)

                    if np.any(dw_x['min temp'].isna()):
                        if np.all(dw_x['min temp'].isna()):
                            dw_x['min temp'] = dw_x['min temp'].fillna(32)
                        else:
                            m = dw['min temp'].mean()
                            dw_x['min temp'] = dw_x['min temp'].fillna(m)

                    if np.any(dw_x['rain'].isna()):
                        dw_x['rain'] = dw_x['rain'].fillna(0)

                    if np.any(dw_x['snow'].isna()):
                        dw_x['snow'] = dw_x['snow'].fillna(0)
                else:
                    continue
            else:
                continue
        else:
            continue

        aq_x = aq_x.join(dw_x).ffill()

        aq_y = aq.loc[ty_range[0]:ty_range[1]].copy(deep=True)
        aq_y.loc[aq_y['pm25'] < 0, 'pm25'] = np.nan
        if (aq_y.isnull().sum()/aq_y.shape[0])[0] < 0.3:
            if aq_y.shape[0] == 24:
                dt = aq_y.index.to_series().diff().dt.seconds.div(3600)
                dt = np.ma.array(dt, mask=np.isnan(dt))
                if np.all(dt == 1):
                    aq_y = aq_y.interpolate('time', limit_direction='both')
                    aq_y.loc[aq_y['pm25'] < 0, 'pm25'] = 0
                else:
                    continue
            else:
                continue
        else:
            continue

        set_x.append(aq_x.to_numpy(copy=True))
        set_y.append(aq_y.to_numpy(copy=True))

set_x = np.array(set_x)
set_y = np.array(set_y)
num = set_x.shape[0]
shuffle = rng.permutation(num)
set_x = set_x[shuffle]
set_y = set_y[shuffle]

for i in range(num):
    set_x[i] = StandardScaler().fit_transform(set_x[i])
    set_y[i] = StandardScaler().fit_transform(set_y[i])

split_1 = int(num*0.8)
split_2 = int(num*0.9)
train_set_x = set_x[:split_1]
train_set_y = set_y[:split_1]
dev_set_x = set_x[split_1:split_2]
dev_set_y = set_y[split_1:split_2]
test_set_x = set_x[split_2:]
test_set_y = set_y[split_2:]

np.save('dataset/train_set_x', train_set_x)
np.save('dataset/train_set_y', train_set_y)
np.save('dataset/dev_set_x', dev_set_x)
np.save('dataset/dev_set_y', dev_set_y)
np.save('dataset/test_set_x', test_set_x)
np.save('dataset/test_set_y', test_set_y)
