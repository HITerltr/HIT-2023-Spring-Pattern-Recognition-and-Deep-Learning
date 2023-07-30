# -*- coding: gbk -*-
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from custom_dataset import Online_Shopping, Jena_Climate
from torch.utils.data import DataLoader


def Load_Jena(path = "Data/Jena_Climate/jena_climate_2009_2016.csv", batch_size = 16, shuffle = True):
    # ��ȡcsv
    csv = pd.read_csv(path, low_memory = False)  # ��ֹ��������
    csv_df = pd.DataFrame(csv)
    temperature_df = csv_df.loc[:, ("Date Time", "T (degC)")]
    temperature_df["datetime"] = [datetime.strptime(x, '%d.%m.%Y %H:%M:%S') for x in temperature_df['Date Time']]
    del temperature_df["Date Time"]

    # ������������
    # Extracting the hour of day
    temperature_df['hour'] = [x.hour for x in temperature_df['datetime']]

    # Extracting the month of the year
    temperature_df['month'] = [x.month for x in temperature_df['datetime']]

    # Creating the cyclical daily feature
    temperature_df['day_cos'] = [np.cos(x * (2 * np.pi / 24)) for x in temperature_df['hour']]
    temperature_df['day_sin'] = [np.sin(x * (2 * np.pi / 24)) for x in temperature_df['hour']]

    # Extracting the timestamp from the datetime object
    temperature_df['timestamp'] = [x.timestamp() for x in temperature_df['datetime']]

    # Seconds in day
    s = 24 * 60 * 60

    # Seconds in year
    year = 365.25 * s

    temperature_df['month_cos'] = [np.cos(x * (2 * np.pi / year)) for x in temperature_df['timestamp']]
    temperature_df['month_sin'] = [np.sin(x * (2 * np.pi / year)) for x in temperature_df['timestamp']]

    # �������ݼ�
    weekly_record = 7 * 24 * 6  # time record of 7 days
    train_record = 5 * 24 * 6
    test_record = 2 * 24 * 6
    boundary = temperature_df.index[temperature_df['datetime'] == '2015-01-01 00:00:00'].tolist()[0]
    del temperature_df["datetime"]
    del temperature_df["hour"]
    del temperature_df["month"]
    del temperature_df["timestamp"]
    # ��һ��
    df_mean = temperature_df.mean()
    df_std = temperature_df.std()
    temperature_df = (temperature_df - df_mean) / df_std

    five_year_df = temperature_df.iloc[:boundary, :]
    two_year_df = temperature_df.iloc[boundary:, :]
    two_year_df.reset_index(drop = True, inplace = True)  # ��������

    def Gen_Train_Test(df):
        known_data = []
        pred_temper = []
        temp_known_data = []
        temp_pred_temper = []

        for index, row in df.iterrows():

            if index % weekly_record < train_record:  # ����Ϊѵ����
                temp_known_data.append(row[:].values)
            else:
                temp_pred_temper.append(row["T (degC)"])
                if index % weekly_record == weekly_record - 1:  # ������һ�֣������һ��
                    temp_known_data_array = np.array(temp_known_data)
                    known_data.append(torch.tensor(temp_known_data_array).float())
                    pred_temper.append(torch.tensor(temp_pred_temper).float())
                    temp_known_data = []
                    temp_pred_temper = []

        return known_data, pred_temper

    train_known_data, train_pred_temper = Gen_Train_Test(five_year_df)
    test_known_data, test_pred_temper = Gen_Train_Test(two_year_df)

    train_dataset = Jena_Climate(train_known_data, train_pred_temper)
    test_dataset = Jena_Climate(test_known_data, test_pred_temper)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = shuffle, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = shuffle, drop_last = True)

    return train_loader, test_loader