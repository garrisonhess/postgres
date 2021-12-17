#!/usr/bin/env python3

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    MultiTaskLasso,
    Lasso,
    ElasticNet,
    MultiTaskElasticNet,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor
import pickle
from config import METHODS, MODEL_DIR


def get_model(method, config):
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")

    regressor = None
    if method == "lr":
        regressor = LinearRegression(n_jobs=config["num_jobs"])
    if method == "huber":
        regressor = HuberRegressor(max_iter=config["huber"]["max_iter"])
        regressor = MultiOutputRegressor(regressor)
    if method == "rf":
        regressor = RandomForestRegressor(
            n_estimators=config["rf"]["n_estimators"],
            criterion=config["rf"]["criterion"],
            n_jobs=config["num_jobs"],
        )
    if method == "gbm":
        regressor = LGBMRegressor(
            max_depth=config["gbm"]["max_depth"],
            num_leaves=config["gbm"]["num_leaves"],
            n_estimators=config["gbm"]["n_estimators"],
            min_child_samples=config["gbm"]["min_child_samples"],
            random_state=config["random_state"],
        )
        regressor = MultiOutputRegressor(regressor)
    if method == "nn":
        hls = tuple(dim for dim in config["mlp"]["hidden_layers"])
        regressor = MLPRegressor(
            hidden_layer_sizes=hls,
            early_stopping=config["mlp"]["early_stopping"],
            max_iter=config["mlp"]["max_iter"],
            alpha=config["mlp"]["alpha"],
        )
    if method == "mt_lasso":
        regressor = MultiTaskLasso(alpha=config["mt_lasso"]["alpha"])
    if method == "lasso":
        regressor = Lasso(alpha=config["lasso"]["alpha"])
    if method == "dt":
        regressor = DecisionTreeRegressor(max_depth=config["dt"]["max_depth"])
        regressor = MultiOutputRegressor(regressor)
    if method == "elastic":
        regressor = ElasticNet(
            alpha=config["elastic"]["alpha"], l1_ratio=config["elastic"]["l1_ratio"]
        )
        regressor = MultiOutputRegressor(regressor)
    if method == "mt_elastic":
        regressor = MultiTaskElasticNet(l1_ratio=config["mt_elastic"]["l1_ratio"])

    return regressor


class BehaviorModel:
    def __init__(self, method, timestamp, config):
        """
        :param method: which ML method to use
        :param normalize: whether to perform standard normalization on data (both x and y)
        :param log_transform: whether to perform log transformation on data (both x and y)
        """

        self.method = method
        self.timestamp = timestamp
        self.model_name = f"{method}_{timestamp}"
        self.model = get_model(method, config)
        self.normalize = config["normalize"]
        self.log_transform = config["log_transform"]
        self.eps = 1e-4
        self.xscaler = RobustScaler() if config["robust"] else StandardScaler()
        self.yscaler = RobustScaler() if config["robust"] else StandardScaler()

    def train(self, x, y):
        if self.log_transform:
            x = np.log(x + self.eps)
            y = np.log(y + self.eps)

        if self.normalize:
            x = self.xscaler.fit_transform(x)
            y = self.yscaler.fit_transform(y)

        self.model.fit(x, y)

    def predict(self, x):
        # transform the features
        if self.log_transform:
            x = np.log(x + self.eps)
        if self.normalize:
            x = self.xscaler.transform(x)

        # make prediction
        y = self.model.predict(x)

        # transform the y back
        if self.normalize:
            y = self.yscaler.inverse_transform(y)
        if self.log_transform:
            y = np.exp(y) - self.eps
            y = np.clip(y, 0, None)

        return y

    def save(self):
        with open(MODEL_DIR / self.model_name, "wb") as f:
            pickle.dump(self.model, f)
