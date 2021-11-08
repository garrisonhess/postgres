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


def load_data(experiment_dir, test_size=0.2):

    ou_name = "ExecIndexScan"
    result_paths = [p for p in experiment_dir.glob("**/*.csv") if ou_name in str(p)]
    print(result_paths)
    df = pd.concat(map(pd.read_csv, result_paths))

    cols_to_remove = ["start_time", "end_time", "cpu_id", "query_id"]
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)
    print(f"Dropped zero-variance columns: {cols_to_remove}")
    print(f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}")

    all_target_cols = [
        "cpu_cycles",
        "instructions",
        "cache_references",
        "cache_misses",
        "ref_cpu_cycles",
        "network_bytes_read",
        "network_bytes_written",
        "disk_bytes_read",
        "disk_bytes_written",
        "memory_bytes",
        "elapsed_us",
    ]
    feat_cols = [col for col in df.columns if col not in all_target_cols]
    target_cols = [col for col in df.columns if col in all_target_cols]

    print(f"OU Name: {ou_name}, Features: {feat_cols}")
    X = df[feat_cols].values
    y = df[target_cols].values
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")

    # bypass train-test split
    if test_size < 0:
        return feat_cols, target_cols, X, X, y, y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return feat_cols, target_cols, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="OU Model Trainer")
    aparser.add_argument("--log", default="info", help="The logging level")
    args = aparser.parse_args()

    # load the data
    experiment_name = "2021-11-08_04-26-47"
    experiment_dir = Path.home() / "postgres/cmudb/tscout/results/tpcc" / experiment_name
    feat_cols, target_cols, X_train, X_test, y_train, y_test = load_data(
        experiment_dir, test_size=0.2
    )

    methods = [
        "lr",
        "rf",
        "gbm",
    ]

    for method in methods:
        # train the model
        ou_model = model.Model(method=method, normalize=True, log_transform=True, robust=True)
        ou_model.train(X_train, y_train)

        # predict
        y_pred = ou_model.predict(X_test)

        print(f"\n============= Model Summary for Model: {method} =============")
        for target_idx, target in enumerate(target_cols):
            target_pred = y_pred[:, target_idx]
            target_true = y_test[:, target_idx]
            mse = mean_squared_error(target_true, target_pred)
            mae = mean_absolute_error(target_true, target_pred)
            exp_var = explained_variance_score(target_true, target_pred)
            r2 = r2_score(target_true, target_pred)
            print(f"===== Target: {target} =====")
            print(f"MSE: {round(mse, 2)}")
            print(f"MAE: {round(mae, 2)}")
            print(f"Explained Variance: {round(exp_var, 2)}")
            print(f"R-Squared: {round(r2, 2)}")
        print("======================== END SUMMARY ========================\n")
