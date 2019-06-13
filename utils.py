import pickle, json
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import math
import seaborn as sns

# from geolib import geohash

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import AxesGrid

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from functools import partial

NUM_LABELS = 5
NUM_REQ = 720 # number of past readings we use (determined by manually tuning)
TIME_FEATURES = ['month', 'day', 'week', 'hour', 'minute', 'dow', 'total_minutes', 'total_minutes_since_day_1']

def prep_holdout_set(holdout_set, df):
    """
    Creates a intermediate dataset from which holdout features can be derived using gen_test_features()
    :param holdout_set: holdout set after adding datetime col
    :param df: transformed training and holdout set
    :return: data (transformed dataset from which to extract holdout features) and geohash_list for indexing
    """
    global time_features
    data = []
    geohash_list = []

    for index, row in holdout_set.iterrows():
        geohash = row['geohash6']
        dt = row['datetime']
        geohash_list.append(geohash)
        # get past X time samples for features (as per gen_features())
        # and next 5 time samples for labels
        data.append(df.loc[geohash][
            pd.date_range(dt - timedelta(minutes=15 * NUM_REQ),
            dt + timedelta(minutes=15 * NUM_LABELS), freq="15min")].values)

    # data is a list of series, each series is a geohash with X number of time samples and T+1 -> T+5 labels
    # different dt for each series
    return data, geohash_list

def gen_test_features(dt, data, geohash_list, dt_list, cluster_df):
    """
    From intermediate holdout set, generate test features the same way train set is generated
    :param dt: arbitrary dt
    :param data: intermediate holdout dataset
    :param geohash_list: generated with intermediate holdout dataset, for indexing
    :param dt_list: list of all start times of each sample in holdout set
    :param cluster_df: cluster features generated in geo eda
    :return: holdout feature set and labels
    """
    X_1, y_1 = gen_features(data, dt, dt_list, is_train=True)

    # add cluster_df
    X_1['geohash6'] = geohash_list
    X_1 = pd.concat([X_1, cluster_df.reindex(X_1['geohash6']).reset_index(drop=True)], axis=1)
    X_1 = X_1.set_index('geohash6')

    return X_1, y_1

def prep_dataset(dt, num_samples, df, cluster_df, is_train=True):
    """
    From transformed dataset, generate combined feature set sampled at different times, and labels
    :param dt: transformed train and holdout set combined
    :param num_samples: number of times to sample, per 15min interval (e.g. 96 is 1 day)
    :param df: transformed train and holdout set combined
    :param cluster_df: cluster features generated in geo eda
    :param is_train: whether this is to be a train set or not
    :return: combined feature set sampled at different times, with labels if train set
    """

    X_1, y_1 = [], []
    for i in range(num_samples):
        delta = timedelta(minutes=15* i)
        # main set
        X_temp, y_temp = gen_features(df, dt - delta, is_train=is_train)

        # combine
        X_temp = pd.concat([X_temp.reset_index(drop=True), cluster_df.reset_index(drop=True)], axis=1)
        X_temp['geohash6'] = df.index.values

        X_1.append(X_temp)
        y_1.append(y_temp)

    X_1 = pd.concat(X_1, axis=0).set_index('geohash6')

    if is_train:
        y_1 = np.concatenate(y_1, axis=0)
        return X_1, y_1
    else:
        return X_1

# Helper functions
# dt is date to sample
def gen_features(df, dt, dt_list=None, is_train=True, name_prefix=None):  # dt is T
    """
    From transformed dataset, generate feature set and labels
    :param df: transformed train and holdout set combined
    :param dt: start time of all samples (arbitrary for holdout set)
    :param dt_list: list of start times, if start time is different for each sample. Else leave as None
    :param is_train: whether is train set or not
    :param name_prefix: unused, used to be for groupby datasets
    :return: feature set and labels
    """

    X = {}

    # mean and other statistics for X period
    for j in range(0, 3):  # for past j hours
        for i in [2, 3, 4, 6, 8]:  # MA of 30mins to 2 hr
            tmp = get_timespan_15(df, dt - timedelta(hours=j) - timedelta(minutes=15 * i), dt)
            X['mean_%s_%s' % (j, i)] = tmp.mean(axis=1).values
            X['diff_%s_%s_mean' % (j, i)] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_%s_decay' % (j, i)] = (tmp * np.power(0.9, np.arange(len(tmp.columns))[::-1])).sum(
                axis=1).values
            X['mean_%s_%s' % (j, i)] = tmp.mean(axis=1).values
            X['median_%s_%s' % (j, i)] = tmp.median(axis=1).values
            X['min_%s_%s' % (j, i)] = tmp.min(axis=1).values
            X['max_%s_%s' % (j, i)] = tmp.max(axis=1).values
            X['std_%s_%s' % (j, i)] = tmp.std(axis=1).values

    X = pd.DataFrame(X)

    # last 15 timings (T -> T-14)
    numTimings = 30 - 1
    last_10_timestamp = get_timespan_15(df, dt - timedelta(minutes=15 * numTimings), dt)
    last_10_timestamp.columns = ["T-{}".format(x) for x in range(numTimings, -1, -1)]
    last_10_timestamp = last_10_timestamp.reset_index(drop=True)

    # 1 week ago (T -> T-14)
    last_10_timestamp_1 = get_timespan_15(df, dt - timedelta(weeks=1) - timedelta(minutes=15 * (numTimings)),
                                          dt - timedelta(weeks=1))
    last_10_timestamp_1.columns = ["W-1_T-{}".format(x) for x in range(numTimings, -1, -1)]
    last_10_timestamp_1 = last_10_timestamp_1.reset_index(drop=True)

    # T+1 -> T+14, 1 week ago
    next_10_timestamp_1 = get_timespan_15(df, dt - timedelta(weeks=1) + timedelta(minutes=15),
                                          dt - timedelta(weeks=1) + timedelta(minutes=15 * (numTimings)))
    next_10_timestamp_1.columns = ["W-1_T+{}".format(x) for x in range(1, numTimings + 1)]
    next_10_timestamp_1 = next_10_timestamp_1.reset_index(drop=True)

    X = pd.concat([X, last_10_timestamp, last_10_timestamp_1, next_10_timestamp_1], axis=1)

    if is_train:
        y = df[dt + timedelta(minutes=15)].values

        # time features
        if(dt_list is None):
            dt_list = pd.Series([dt]*len(df.index))
        temp = dt_list.apply(extract_time)
        df_time = pd.DataFrame(temp.values.tolist(), columns=TIME_FEATURES)
        X =  pd.concat([X,df_time], axis=1)

        return X, y

    if name_prefix is not None:
        X.columns = ['%s_%s' % (name_prefix, c) for c in X.columns]
    return X, None

# Extract time values from given timestamp
# Note: output order has to match TIME_FEATURES
def extract_time(dt):
    month, day = int(dt.strftime("%m")), int(dt.strftime("%d"))
    hour, minute = int(dt.strftime("%H")), int(dt.strftime("%M"))
    total_minutes, total_minutes_since_day_1 = minute + hour * 60, minute + hour * 60 + day * 24 * 60 + month * 30 * 24 * 60
    week, dow = math.floor((day + (month - 4) * 30) / 7), (day - 1) % 7
    return month, day, week, hour, minute, dow, total_minutes, total_minutes_since_day_1

# get range of demand values for specified time range
def get_timespan_15(df, startTime, endTime, freq='15min'):  # inclusive
    return df[pd.date_range(startTime, endTime, freq=freq)]

# ========================================================
# Preprocessing

# Parse datetime into separate columns, using arbitrary start date, modifies in place
def parse_datetime(df, start_date):
    # Convert 'day' and 'timestamp' cols to datetime
    df[['hour', 'minute']] = df['timestamp'].str.split(':', expand=True)
    df['datetime'] = df['day'].apply(lambda x: start_date + pd.to_timedelta(x, unit='D') - timedelta(days=1))
    df['day'] = df['datetime'].apply(lambda x: int(x.strftime('%d')))
    df['month'] = df['datetime'].apply(lambda x: int(x.strftime('%m')))
    df['dow'] = (df['day'] - 1) % 7

    # require unique for index
    df['datetime'] = pd.to_datetime(dict(year=2019, month=df.month, day=df.day, hour=df.hour, minute=df.minute))


# Transform dataset such that dates are the columns, rows are the locations and cells are the demand values.
# Also fill missing dates.
def transform_df(df, start_date, num_days):
    df_transformed = df.set_index(
        ["geohash6", "datetime"])[["demand"]].unstack(
        level=-1).fillna(0)
    df_transformed.columns = df_transformed.columns.get_level_values(1)

    # fill in missing timestamp cols (all on 18/4)
    start_date = start_date
    end_date = start_date + timedelta(days=num_days)
    missing_cols = sorted(list(
        set(pd.date_range(start_date, end_date, freq='15min').values).difference(set(df_transformed.columns.values))))[
                   :-1]  # get rid of 6/1

    missing_dates = {}
    for i in missing_cols:
        missing_dates[i] = [0] * len(df_transformed.index)
    missing_dates = pd.DataFrame(missing_dates)
    missing_dates.index = df_transformed.index

    df_transformed = pd.concat([df_transformed, missing_dates], axis=1)
    df_transformed = df_transformed.reindex(sorted(df_transformed.columns), axis=1)
    df_transformed.columns = pd.DatetimeIndex(df_transformed.columns)

    return df_transformed
