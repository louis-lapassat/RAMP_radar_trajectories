import numpy as np

# data matrix indices
[T, X, Vx, Ax, Jx, Y, Vy, Ay, Jy, Z, Vz, Az, Jz, U2, C2, U3, C3, T3] =\
    range(18)


class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        data_matrix = np.asarray(list(X_df['data'].values))
        # taking the mean of each data matrix variable
        means = data_matrix.mean(axis=1)
        return means
