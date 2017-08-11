import numpy as np


class PlaneModelND:
    def __init__(self):
        self.params = None

    def estimate(self, data):
        if data.shape[0] > 2:  # well determined
            origin = data.mean(axis=0)
            data_adjust = data - origin
            matrix = np.cov(data_adjust.T)
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            sort = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, sort]
            direction = eigenvectors[-1]
            self.params = (origin, direction)
        else:  # under-determined
            raise ValueError('At least 3 input points needed.')
        return True

    def residuals(self, data, params=None):
        if params is None:
            params = self.params
        assert params is not None
        if len(params) != 2:
            raise ValueError('Parameters are defined by 2 sets.')
        origin, direction = self.params
        res = np.dot(data-origin,direction)/np.linalg.norm(direction)[..., np.newaxis]
        return np.absolute(res)
