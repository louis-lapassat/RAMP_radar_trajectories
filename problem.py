import os
import numpy as np
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = 'Aircraft classification from radar trajectories'
_target_column_name = 'object_type'
_prediction_label_names = [
    '1111', '1112', '1121', '1122', '1132', '1222', '1224', '1231',
    '1232', '1233', '1234', '1324', '1332', '1333', '1334', '4111',
    '4121', '4122', '4222']
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

score_types = [
    rw.score_types.NegativeLogLikelihood(name='nll', precision=3),
    rw.score_types.Accuracy(name='acc', precision=3),
]


def get_cv(X, y):
    X['object_type'] = y
    X['unique_trajectory_id'] = X['object_type'].astype(str) + '_'\
        + X['trajectory_id'].astype(str)
    trajectories = X.groupby(['unique_trajectory_id']).sum().reset_index()
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=51)
    r = np.arange(len(X))
    for train_trajectory_is, test_trajectory_is\
            in cv.split(trajectories, trajectories['object_type']):
        train_is = r[np.in1d(
            X['unique_trajectory_id'],
            trajectories['unique_trajectory_id'][train_trajectory_is].values)]
        test_is = r[np.in1d(
            X['unique_trajectory_id'],
            trajectories['unique_trajectory_id'][test_trajectory_is].values)]
        yield train_is, test_is


def _read_data(path, f_name):
    data = pd.read_pickle(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, 'f_name', 'start_t'], axis=1)
    X_df['index'] = range(len(X_df))
    X_df = X_df.set_index('index')
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.pkl'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.pkl'
    return _read_data(path, f_name)
