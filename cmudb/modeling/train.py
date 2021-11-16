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
import itertools
import model

np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=10)
np.set_printoptions(suppress=True)

BENCHMARK_NAMES = ["tpcc", "tpch", "ycsb", "wikipedia", "voter", "twitter", "tatp", "smallbank", "sibench", "seats", "resourcestresser", "noop", "hyadapt", "epinions", "chbenchmark", "auctionmark"]

OU_NAMES = [
    "ExecAgg",
    "ExecAppend",
    "ExecCteScan",
    "ExecCustomScan",
    "ExecForeignScan",
    "ExecFunctionScan",
    "ExecGather",
    "ExecGatherMerge",
    "ExecGroup",
    "ExecHashJoinImpl",
    "ExecIncrementalSort",
    "ExecIndexOnlyScan",
    "ExecIndexScan",
    "ExecLimit",
    "ExecLockRows",
    "ExecMaterial",
    "ExecMergeAppend",
    "ExecMergeJoin",
    "ExecModifyTable",
    "ExecNamedTuplestoreScan",
    "ExecNestLoop",
    "ExecProjectSet",
    "ExecRecursiveUnion",
    "ExecResult",
    "ExecSampleScan",
    "ExecSeqScan",
    "ExecSetOp",
    "ExecSort",
    "ExecSubPlan",
    "ExecSubqueryScan",
    "ExecTableFuncScan",
    "ExecTidScan",
    "ExecUnique",
    "ExecValuesScan",
    "ExecWindowAgg",
    "ExecWorkTableScan"]


class OUModelTrainer:
    """
    Trainer for the ou models
    """

    def __init__(self, input_path):
        self.input_path = input_path
        self.stats_map = {}

    def train(self):
        pass


def load_data(experiment_dir):
    result_paths = [fp for fp in experiment_dir.glob("**/*.csv")]
    ou_name_to_df = dict()
    ou_name_to_nruns = dict()

    for ou_name in OU_NAMES:
        ou_results = [fp for fp in result_paths if fp.name == f"{ou_name}.csv" and os.stat(fp).st_size > 0]
        if len(ou_results) > 0: 
            # print(f"Found {len(ou_results)} run(s) for {ou_name}")
            ou_name_to_nruns[ou_name] = len(ou_results)
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))
    
    return ou_name_to_df, ou_name_to_nruns


def prep_data(df, test_size=0.2):
    cols_to_remove = ["start_time", "end_time", "cpu_id", "query_id"]
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)
    # print(f"Dropped zero-variance columns: {cols_to_remove}")
    # print(f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}")

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

    X = df[feat_cols].values
    y = df[target_cols].values

    # bypass train-test split
    if test_size < 0:
        return feat_cols, target_cols, X, X, y, y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return feat_cols, target_cols, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OU Model Trainer")
    parser.add_argument("--log", default="info", help="The logging level")
    parser.add_argument("--benchmark-name", default="tpcc", help=f"Benchmarks include: {BENCHMARK_NAMES}")
    parser.add_argument("--experiment-name", required=False, help="Experiment Name")
    args = parser.parse_args()
    benchmark_name = args.benchmark_name
    experiment_name = args.experiment_name

    if benchmark_name not in BENCHMARK_NAMES:
        raise Exception(f"Benchmark name {benchmark_name} not supported")
    
    benchmark_results_dir = Path.home() / "postgres/cmudb/tscout/results/" / benchmark_name

    # if no experiment name is provided, try to find one
    if experiment_name is None:
        experiment_list = sorted([exp_path.name for exp_path in benchmark_results_dir.glob("*")])
        print(f"{benchmark_name} experiments: {experiment_list}")

        if len(experiment_list) == 0:
            raise Exception(f"No experiments found for {benchmark_name}")
        
        experiment_name = experiment_list[-1]
        print(f"Experiment name was not provided, using experiment: {experiment_name}")

    experiment_dir = benchmark_results_dir / experiment_name
    evaluation_dir = Path.home() / "postgres/cmudb/modeling/evaluation"  / benchmark_name / experiment_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["lr", "rf", "gbm"]
    ou_name_to_df, ou_name_to_nruns = load_data(experiment_dir)
    
    for ou_name in ou_name_to_df.keys(): 
        feat_cols, target_cols, X_train, X_test, y_train, y_test = prep_data(ou_name_to_df[ou_name], test_size=0.2)
        ou_eval_dir = evaluation_dir / ou_name
        ou_eval_dir.mkdir(exist_ok=True)

        if X_train.shape[1] == 0 or y_train.shape[1] == 0:
            print(f"{ou_name} has no valid training data, skipping")
            continue

        for method in methods:
            ou_model = model.Model(
                method=method, normalize=True, log_transform=False, robust=False
            )
            ou_model.train(X_train, y_train)
            y_train_pred = ou_model.predict(X_train)
            y_test_pred = ou_model.predict(X_test)

            # pair and reorder the target columns for readable outputs
            paired_cols = zip(target_cols, [f"pred_{col}" for col in target_cols])
            reordered_cols = feat_cols + list(itertools.chain.from_iterable(paired_cols))

            train_preds_path = ou_eval_dir / f"{ou_name}_{method}_train_preds.csv"
            with open(train_preds_path, "w+") as train_preds_file:
                temp = np.concatenate((X_train, y_train, y_train_pred), axis=1)
                train_result_df = pd.DataFrame(temp, columns=feat_cols + target_cols + [f"pred_{col}" for col in target_cols])
                train_result_df[reordered_cols].to_csv(train_preds_file, float_format="%.1f", index=False)
            
            test_preds_path = ou_eval_dir / f"{ou_name}_{method}_test_preds.csv"
            with open(test_preds_path, "w+") as test_preds_file:
                temp = np.concatenate((X_test, y_test, y_test_pred), axis=1)
                test_result_df = pd.DataFrame(temp, columns=feat_cols + target_cols + [f"pred_{col}" for col in target_cols])
                test_result_df[reordered_cols].to_csv(test_preds_file, float_format="%.1f", index=False)

            ou_eval_path = ou_eval_dir / f"{ou_name}_{method}_summary.txt"
            with open(ou_eval_path, "w+") as eval_file:
                eval_file.write(f"\n============= Model Summary for {ou_name} Model: {method} =============\n")
                eval_file.write(f"Num Runs used: {ou_name_to_nruns[ou_name]}\n")
                eval_file.write(f"Features used: {feat_cols}\n")
                eval_file.write(f"Num Features used: {len(feat_cols)}\n")
                eval_file.write(f"Targets estimated: {target_cols}\n")

                for target_idx, target in enumerate(target_cols):
                    eval_file.write(f"===== Target: {target} =====\n")
                    train_target_pred = y_train_pred[:, target_idx]
                    train_target_true = y_train[:, target_idx]
                    mse = mean_squared_error(train_target_true, train_target_pred)
                    mae = mean_absolute_error(train_target_true, train_target_pred)
                    r2 = r2_score(train_target_true, train_target_pred)
                    eval_file.write(f"Train MSE: {round(mse, 2)}\n")
                    eval_file.write(f"Train MAE: {round(mae, 2)}\n")
                    eval_file.write(f"Train R^2: {round(r2, 2)}\n")

                    test_target_pred = y_test_pred[:, target_idx]
                    test_target_true = y_test[:, target_idx]
                    mse = mean_squared_error(test_target_true, test_target_pred)
                    mae = mean_absolute_error(test_target_true, test_target_pred)
                    r2 = r2_score(test_target_true, test_target_pred)
                    eval_file.write(f"Test MSE: {round(mse, 2)}\n")
                    eval_file.write(f"Test MAE: {round(mae, 2)}\n")
                    eval_file.write(f"Test R^2: {round(r2, 2)}\n")
                eval_file.write("======================== END SUMMARY ========================\n")
