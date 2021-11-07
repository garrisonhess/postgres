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

    df = ou_name_to_df["ExecIndexScan"]
    cols_to_remove = []
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)
    print(f"Dropped zero-variance columns: {cols_to_remove}")
    print(f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}")

    df = df.drop(["start_time", "end_time"], axis=1)

    X = df[
        [
            "IndexScanState_iss_NumScanKeys",
            "IndexScanState_iss_NumRuntimeKeys",
            "IndexScanState_iss_RuntimeKeysReady",
            "Plan_startup_cost",
            "Plan_total_cost",
        ]
    ].values

    y = df[
        ["cpu_cycles", "instructions", "cache_references", "cache_misses", "elapsed_us"]
    ].values

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
    X_train, X_test, y_train, y_test = load_data(results_dir, test_size=0.1)
    methods = [
        "lr",
        "huber",
        "rf",
        "gbm",
        "mt_lasso",
        "lasso",
        "dt",
        "mt_elastic",
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
        ou_model = model.Model(method=method, normalize=True, log_transform=False)
        ou_model.train(X_train, y_train)

        # predict
        y_pred = ou_model.predict(X_test)

        print(f"============= Model Summary for Model: {method} =============")
        for target_idx, target in enumerate(targets):
            print(y_pred.shape)
            print(y_test.shape)
            target_pred = y_pred[:, target_idx]
            target_true = y_test[:, target_idx]
            mse = mean_squared_error(target_pred, target_true)
            exp_var = explained_variance_score(target_pred, target_true)
            mae = mean_absolute_error(target_pred, target_true)
            r2 = r2_score(target_pred, target_true)
            print(f"===== Target: {target} =====")
            print(f"MSE: {mse}")
            print(f"MAE: {mae}")
            print(f"Explained Variance: {exp_var}")
            print(f"R-Squared: {r2}")

        print("======================== END SUMMARY ========================")
