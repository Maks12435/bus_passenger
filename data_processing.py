import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S').dt.time
    data['departure_time'] = pd.to_datetime(data['departure_time'], format='%H:%M:%S').dt.time

    def time_diff(row, data):
        current_index = row.name
        next_index = current_index + 1
        if next_index < len(data):
            dt1 = datetime.combine(datetime.today(), row['departure_time'])
            dt2 = datetime.combine(datetime.today(), data.loc[next_index, 'arrival_time'])
            x = (dt2 - dt1).total_seconds()
            return x if 0 <= x <= 1000 else pd.NaT
        else:
            return pd.NaT

    data['travel_time'] = data.apply(time_diff, axis=1, args=(data,))
    data['day_of_week'] = data['date'].dt.dayofweek
    data['hour'] = data['arrival_time'].apply(lambda x: x.hour)
    data['dwell_time_in_seconds'] = data['dwell_time_in_seconds'].astype(float)
    data['total_travel_time'] = data['travel_time'] + data['dwell_time_in_seconds']
    data.drop(['date', 'arrival_time', 'departure_time'], axis=1, inplace=True)
    data.dropna(inplace=True)

    return data
