#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import model

np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=10)
np.set_printoptions(suppress=True)


class OUModelTrainer:
    """
    Trainer for the ou models
    """

    def __init__(
        self, input_path,
    ):
        self.input_path = input_path
        self.stats_map = {}

    def train(self):
        pass


def load_data(results_dir, test_size=0.2):
    run_dirs = sorted(os.listdir(results_dir), reverse=True)
    print(f"Runs: {run_dirs}")
    run_timestamp = run_dirs[0]
    run_path = Path(results_dir) / run_timestamp
    print(f"Taking latest run, recorded at {run_timestamp} UTC")

    ou_name_to_df = dict()

    for filename in os.listdir(run_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(run_path, filename)
        ou_name = filename.split(".")[0]

        if os.stat(filepath).st_size > 0:
            ou_name_to_df[ou_name] = pd.read_csv(filepath)

    for (ou_name, ou_df) in ou_name_to_df.items():
        print(f"OU Name: {ou_name}, ou_df shape: {ou_df.shape}")

    ou_name = "ExecIndexScan"
    df = ou_name_to_df[ou_name]

    # drop features we don't want for now
    features_to_drop = ["start_time", "end_time", "cpu_id", "query_id"]
    df = df.drop(features_to_drop, axis=1)

    cols_to_remove = []
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)
    # print(f"Dropped zero-variance columns: {cols_to_remove}")
    # print(f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}")

    targets = [
        "cpu_cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "elapsed_us",
    ]
    features = [col for col in df.columns if col not in targets]

    print(f"OU Name: {ou_name}, Features: {features}")
    X = df[features].values
    y = df[targets].values

    # bypass train-test split
    if test_size < 0:
        return X, X, y, y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="OU Model Trainer")
    aparser.add_argument("--log", default="info", help="The logging level")
    args = aparser.parse_args()

    # load the data
    results_dir = Path.home() / "postgres/cmudb/tscout/results/tpcc"
    X_train, X_test, y_train, y_test = load_data(results_dir, test_size=-1)
    methods = [
        "lr",
        "huber",
        "rf",
        "gbm",
    ]

    targets = [
        "cpu_cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "elapsed_us",
    ]

    for method in methods:
        # train the model
        ou_model = model.Model(
            method=method, normalize=True, log_transform=True, robust=True
        )
        ou_model.train(X_train, y_train)

        # predict
        y_pred = ou_model.predict(X_test)

        print(f"\n============= Model Summary for Model: {method} =============")
        for target_idx, target in enumerate(targets):
            target_pred = y_pred[:, target_idx]
            target_true = y_test[:, target_idx]
            mse = mean_squared_error(target_pred, target_true)
            exp_var = explained_variance_score(target_pred, target_true)
            mae = mean_absolute_error(target_pred, target_true)
            r2 = r2_score(target_pred, target_true)
            print(f"===== Target: {target} =====")
            print(f"MSE: {round(mse, 2)}")
            print(f"MAE: {round(mae, 2)}")
            print(f"Explained Variance: {round(exp_var, 2)}")
            print(f"R-Squared: {round(r2, 2)}")
        print("======================== END SUMMARY ========================\n")
