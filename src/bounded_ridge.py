"""Ridge regression with non-negative coefficients for media features.

Advertising should never reduce sales, so the first `n_positive` coefficients
(the media features) are constrained to be >= 0. External factors (price,
competitors, economy, ...) can still be positive or negative.

Drop-in replacement for sklearn's Ridge inside the modelling Pipeline:
it exposes `alpha`, `coef_`, `intercept_`, `fit` and `predict`, so
GridSearchCV, joblib, MMMPredictor and the ROI/optimisation notebooks
work without changes.
"""
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class NonNegativeMediaRidge(RegressorMixin, BaseEstimator):
    """Ridge regression where the first `n_positive` coefficients must be >= 0.

    Parameters
    ----------
    alpha : float
        L2 regularisation strength (same meaning as sklearn's Ridge).
    n_positive : int
        Number of leading features constrained to be non-negative
        (the media features come first in the feature list).
    """

    def __init__(self, alpha=1.0, n_positive=0):
        self.alpha = alpha
        self.n_positive = n_positive

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_features = X.shape[1]

        # centre the data so the intercept is not penalised (as in sklearn's Ridge)
        x_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - x_mean
        yc = y - y_mean

        # ridge as a least-squares problem: [X; sqrt(alpha) I] b = [y; 0]
        A = np.vstack([Xc, np.sqrt(self.alpha) * np.eye(n_features)])
        b = np.concatenate([yc, np.zeros(n_features)])

        lower = np.full(n_features, -np.inf)
        lower[: self.n_positive] = 0.0
        upper = np.full(n_features, np.inf)

        result = lsq_linear(A, b, bounds=(lower, upper), method="bvls")

        self.coef_ = result.x
        self.intercept_ = y_mean - x_mean @ self.coef_
        self.n_features_in_ = n_features
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
