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
        V3 = data_matrix[:, :, [Vx, Vy, Vz]]
        v3 = np.linalg.norm(V3, axis=2)
        V2 = data_matrix[:, :, [Vx, Vy]]
        v2 = np.linalg.norm(V2, axis=2)
        new_features = np.reshape(
            [v2, v3], newshape=(
                data_matrix.shape[0], data_matrix.shape[1], -1))
        data_matrix = np.concatenate((data_matrix, new_features), axis=2)
        # taking the mean of each data matrix variable
        means = data_matrix.mean(axis=1)
        stds = data_matrix.std(axis=1)
        return np.concatenate((means, stds), axis=1)
