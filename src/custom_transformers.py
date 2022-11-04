import numpy as np
import numpy.typing as npt


class CreateTotalIncome:
    def fit(self, X, y=None):
        return self

    def transform(self, X: npt.NDArray):
        total_income = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        X = np.concatenate((X, total_income), axis=1)
        return total_income.reshape(-1, 1)
