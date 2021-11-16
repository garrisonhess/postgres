#!/usr/bin/env python3

import numpy as np
import lightgbm as lgb
from sklearn import (
    linear_model,
    kernel_ridge,
    ensemble,
    preprocessing,
    neural_network,
    multioutput,
    svm,
    tree,
)

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)

_LOGTRANS_EPS = 1e-4

METHODS = [
    "lr",
    "huber",
    "svr",
    "kr",
    "rf",
    "gbm",
    "nn",
    "mt_lasso",
    "lasso",
    "dt",
    "mt_elastic",
]


def _get_base_ml_model(method):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    regressor = None
    if method == "lr":
        regressor = linear_model.LinearRegression(n_jobs=8)
    if method == "huber":
        regressor = linear_model.HuberRegressor(max_iter=50)
        regressor = multioutput.MultiOutputRegressor(regressor)
    if method == "svr":
        regressor = svm.LinearSVR()
        regressor = multioutput.MultiOutputRegressor(regressor)
    if method == "kr":
        regressor = kernel_ridge.KernelRidge(kernel="rbf", n_jobs=8)
    if method == "rf":
        regressor = ensemble.RandomForestRegressor(n_estimators=50, n_jobs=8)
    if method == "gbm":
        regressor = lgb.LGBMRegressor(
            max_depth=31,
            num_leaves=1000,
            n_estimators=100,
            min_child_samples=5,
            random_state=42,
        )
        regressor = multioutput.MultiOutputRegressor(regressor)
    if method == "nn":
        regressor = neural_network.MLPRegressor(
            hidden_layer_sizes=(25, 25), early_stopping=True, max_iter=1000000, alpha=5
        )
    if method == "mt_lasso":
        regressor = linear_model.MultiTaskLasso(alpha=1.0)
    if method == "lasso":
        regressor = linear_model.Lasso(alpha=1.0)
    if method == "dt":
        regressor = tree.DecisionTreeRegressor()
    if method == "mt_elastic":
        regressor = linear_model.MultiTaskElasticNet(l1_ratio=0.5)

    return regressor


class Model:
    """
    The class that wraps around standard ML libraries.
    With the implementation for different normalization handlings
    """

    def __init__(self, method, normalize=True, log_transform=True, robust=False):
        """
        :param method: which ML method to use
        :param normalize: whether to perform standard normalization on data (both x and y)
        :param log_transform: whether to perform log transformation on data (both x and y)
        """
        self._base_model = _get_base_ml_model(method)
        self._normalize = normalize
        self._log_transform = log_transform
        self._xscaler = (
            preprocessing.StandardScaler()
            if not robust
            else preprocessing.RobustScaler()
        )
        self._yscaler = (
            preprocessing.StandardScaler()
            if not robust
            else preprocessing.RobustScaler()
        )

    def train(self, x, y):
        if self._log_transform:
            x = np.log(x + _LOGTRANS_EPS)
            y = np.log(y + _LOGTRANS_EPS)

        if self._normalize:
            x = self._xscaler.fit_transform(x)
            y = self._yscaler.fit_transform(y)

        self._base_model.fit(x, y)

    def predict(self, x):
        # transform the features
        if self._log_transform:
            x = np.log(x + _LOGTRANS_EPS)
        if self._normalize:
            x = self._xscaler.transform(x)

        # make prediction
        y = self._base_model.predict(x)

        # transform the y back
        if self._normalize:
            y = self._yscaler.inverse_transform(y)
        if self._log_transform:
            y = np.exp(y) - _LOGTRANS_EPS
            y = np.clip(y, 0, None)

        return y
