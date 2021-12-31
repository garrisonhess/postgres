#!/usr/bin/env python3

import argparse
import itertools
import os
from datetime import datetime

import yaml

import numpy as np
import pandas as pd
import pydotplus
from config import BENCH_DBS, EVAL_DATA_ROOT, MODEL_CONFIG_DIR, MODEL_DIR, OU_NAMES, TRAIN_DATA_ROOT, logger
from model import BehaviorModel
from sklearn import tree
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

BASE_TARGET_COLS = [
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

DIFF_TARGET_COLS = [
    "diffed_cpu_cycles",
    "diffed_instructions",
    "diffed_cache_references",
    "diffed_cache_misses",
    "diffed_ref_cpu_cycles",
    "diffed_network_bytes_read",
    "diffed_network_bytes_written",
    "diffed_disk_bytes_read",
    "diffed_disk_bytes_written",
    "diffed_memory_bytes",
    "diffed_elapsed_us",
]

ALL_TARGET_COLS = BASE_TARGET_COLS + DIFF_TARGET_COLS


def evaluate(ou_model, X, y, output_dir, dataset, mode):
    if mode != "train" and mode != "eval":
        raise ValueError(f"Invalid mode: {mode}")

    if mode == "train":
        pretty_mode = "Training"
    else:
        pretty_mode = "Evaluation"

    y_pred = ou_model.predict(X)

    # pair and reorder the target columns for readable outputs
    paired_cols = zip([f"pred_{col}" for col in target_cols], target_cols)
    reordered_cols = feat_cols + list(itertools.chain.from_iterable(paired_cols))

    preds_path = output_dir / f"{ou_model.ou_name}_{ou_model.method}_{dataset}_{pretty_mode}_preds.csv"
    with open(preds_path, "w+") as preds_file:
        temp = np.concatenate((X, y, y_pred), axis=1)
        test_result_df = pd.DataFrame(temp, columns=feat_cols + target_cols + [f"pred_{col}" for col in target_cols])
        test_result_df[reordered_cols].to_csv(preds_file, float_format="%.1f", index=False)

    if ou_model.method == "dt" and mode == "train":
        for idx, target_name in enumerate(target_cols):
            dot = tree.export_graphviz(ou_model.model.estimators_[idx], feature_names=feat_cols, filled=True)
            dt_file = f"{output_dir}/{ou_name}_{pretty_mode}_treeplot_{target_name}.png"
            pydotplus.graphviz.graph_from_dot_data(dot).write_png(dt_file)

    ou_eval_path = output_dir / f"{ou_model.ou_name}_{ou_model.method}_{dataset}_{pretty_mode}_summary.txt"
    with open(ou_eval_path, "w+") as eval_file:
        eval_file.write(f"\n============= {pretty_mode}: Model Summary for {ou_name} Model: {method} =============\n")
        eval_file.write(f"Features used: {feat_cols}\n")
        eval_file.write(f"Num Features used: {len(feat_cols)}\n")
        eval_file.write(f"Targets estimated: {target_cols}\n")

        for target_idx, target in enumerate(target_cols):
            eval_file.write(f"===== Target: {target} =====\n")
            target_pred = y_pred[:, target_idx]
            target_true = y[:, target_idx]
            true_mean = target_true.mean()
            pred_mean = target_pred.mean()
            mse = mean_squared_error(target_true, target_pred)
            mae = mean_absolute_error(target_true, target_pred)
            mape = mean_absolute_percentage_error(target_true, target_pred)
            r2 = r2_score(target_true, target_pred)
            eval_file.write(f"Target Mean: {round(true_mean, 2)}, Predicted Mean: {round(pred_mean, 2)}\n")
            eval_file.write(f"Mean Absolute Percentage Error (MAPE): {round(mape, 2)}\n")
            eval_file.write(f"Mean Squared Error (MSE): {round(mse, 2)}\n")
            eval_file.write(f"Mean Absolute Error (MAE): {round(mae, 2)}\n")
            eval_file.write(f"Percentage Explained Variation (R-squared): {round(r2, 2)}\n")

        eval_file.write("======================== END SUMMARY ========================\n")


def load_data(data_dir):
    result_paths = [fp for fp in data_dir.glob("*.csv") if os.stat(fp).st_size > 0]
    ou_name_to_df = dict()

    for ou_name in OU_NAMES:
        ou_results = [fp for fp in result_paths if fp.name.startswith(ou_name)]
        if len(ou_results) > 0:
            logger.info(f"Found {len(ou_results)} run(s) for {ou_name}")
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))

    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in data_dir: {data_dir}")

    return ou_name_to_df


def prep_train_data(df, feat_diff, target_diff):

    cols_to_remove = ["start_time", "end_time", "cpu_id", "query_id", "rid", "plan_node_id"]

    if target_diff:
        cols_to_remove += BASE_TARGET_COLS
    else:
        cols_to_remove += DIFF_TARGET_COLS

    cols_to_remove = [x for x in cols_to_remove if x in df.columns]

    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)
    df = df.sort_index(axis=1)

    if len(cols_to_remove) > 0:
        logger.info(f"Dropped zero-variance columns: {cols_to_remove}")
        logger.info(f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}")

    if target_diff:
        target_cols = [col for col in df.columns if col in DIFF_TARGET_COLS]
    else:
        target_cols = [col for col in df.columns if col in BASE_TARGET_COLS]

    feat_cols = [col for col in df.columns if col not in ALL_TARGET_COLS]

    if not feat_diff:
        feat_cols = [col for col in feat_cols if not col.startswith("diffed")]

    X = df[feat_cols].values
    y = df[target_cols].values

    return feat_cols, target_cols, X, y


def prep_eval_data(df, feat_cols, target_cols):
    X = df[feat_cols].values
    y = df[target_cols].values

    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OU Model Trainer")
    parser.add_argument("--config_name", type=str, default="default")
    args = parser.parse_args()
    config_name = args.config_name

    # load config
    config_path = MODEL_CONFIG_DIR / f"{config_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file: {config_name} does not exist")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_bench_dbs = config["train_bench_dbs"]
    train_bench_db = train_bench_dbs[0]
    eval_bench_dbs = config["eval_bench_dbs"]
    eval_bench_db = eval_bench_dbs[0]
    feat_diff = config["features_diff"]
    target_diff = config["targets_diff"]

    for train_bench_db in train_bench_dbs:
        if train_bench_db not in BENCH_DBS:
            raise ValueError(f"Benchmark DB {config['bench_db']} not supported")

    # if no experiment name is provided, try to find one
    if config["experiment_name"] is None:
        experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
        logger.warning(f"{train_bench_db} experiments: {experiment_list}")
        assert len(experiment_list) > 0, "No experiments found"
        experiment_name = experiment_list[-1]
        logger.warning(f"Experiment name was not provided, using experiment: {experiment_name}")

    training_data_dir = TRAIN_DATA_ROOT / experiment_name / train_bench_db
    eval_data_dir = EVAL_DATA_ROOT / experiment_name / eval_bench_db
    logger.warning(f"eval data dir: {eval_data_dir}")
    if not training_data_dir.exists():
        raise ValueError(f"Train Benchmark DB {train_bench_db} not found in experiment: {experiment_name}")
    if not eval_data_dir.exists():
        raise ValueError(f"Eval Benchmark DB {eval_bench_db} not found in experiment: {experiment_name}")

    train_ou_to_df = load_data(training_data_dir)
    eval_ou_to_df = load_data(eval_data_dir)
    base_model_name = f"{config_name}_{training_timestamp}"
    output_dir = MODEL_DIR / base_model_name

    for ou_name in train_ou_to_df.keys():
        logger.warning(f"Begin Training OU: {ou_name}")
        feat_cols, target_cols, X_train, y_train = prep_train_data(train_ou_to_df[ou_name], feat_diff, target_diff)
        ou_model_name = f"{ou_name}_{base_model_name}"

        if X_train.shape[1] == 0 or y_train.shape[1] == 0:
            logger.warning(f"{ou_name} has no valid training data, skipping")
            continue

        for method in config["methods"]:
            full_outdir = output_dir / method / ou_name
            full_outdir.mkdir(parents=True, exist_ok=True)
            logger.warning(f"Training OU: {ou_name} with model: {method}")
            ou_model = BehaviorModel(method, ou_name, base_model_name, config, feat_cols, target_cols)
            ou_model.train(X_train, y_train)
            ou_model.save()
            evaluate(ou_model, X_train, y_train, full_outdir, train_bench_db, mode="train")
            X_eval, y_eval = prep_eval_data(eval_ou_to_df[ou_name], feat_cols, target_cols)
            evaluate(ou_model, X_eval, y_eval, full_outdir, eval_bench_db, mode="eval")
