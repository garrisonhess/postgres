import argparse
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import model_selection

import model

# from . import model

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


def load_data(results_dir):
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

    features = df[
        [
            "IndexScanState_iss_NumScanKeys",
            "IndexScanState_iss_NumRuntimeKeys",
            "IndexScanState_iss_RuntimeKeysReady",
            "Plan_startup_cost",
            "Plan_total_cost",
            "cpu_id",
        ]
    ]

    targets = df[
        ["cpu_cycles", "instructions", "cache_references", "cache_misses", "elapsed_us"]
    ]

    return features, targets


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="OU Model Trainer")
    aparser.add_argument("--log", default="info", help="The logging level")
    args = aparser.parse_args()

    # load the data
    results_dir = Path.home() / "postgres/cmudb/tscout/results/tpcc"
    features, targets = load_data(results_dir)

    # train the model
    rf_model = model.Model(method="rf", normalize=False, log_transform=False)
    rf_model.train(features, targets)

    # predict
    preds = rf_model.predict(features)
    print(preds)

    # # evaluate

