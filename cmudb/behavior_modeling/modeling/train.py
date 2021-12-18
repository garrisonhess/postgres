#!/usr/bin/env python3

import argparse
from datetime import datetime
import os
import pandas as pd
import yaml
from model import BehaviorModel
from behavior_modeling import OU_NAMES, BENCH_DBS, TARGET_COLS, CONFIG_DIR, EVAL_DIR, DATA_DIR, logger

def load_data(experiment_dir):
    result_paths = [fp for fp in experiment_dir.glob("**/*.csv")]
    ou_name_to_df = dict()
    ou_name_to_nruns = dict()

    for ou_name in OU_NAMES:
        ou_results = [
            fp
            for fp in result_paths
            if fp.name == f"{ou_name}.csv" and os.stat(fp).st_size > 0
        ]
        if len(ou_results) > 0:
            logger.info(f"Found {len(ou_results)} run(s) for {ou_name}")
            ou_name_to_nruns[ou_name] = len(ou_results)
            ou_name_to_df[ou_name] = pd.concat(map(pd.read_csv, ou_results))

    if len(ou_name_to_df) == 0:
        raise Exception(f"No data found in experiment_dir: {experiment_dir}")

    return ou_name_to_df, ou_name_to_nruns


def prep_data(df):
    cols_to_remove = ["start_time", "end_time", "cpu_id", "query_id"]
    for col in df.columns:
        if df[col].nunique() == 1:
            cols_to_remove.append(col)

    df = df.drop(cols_to_remove, axis=1)

    if len(cols_to_remove) > 0:
        logger.info(f"Dropped zero-variance columns: {cols_to_remove}")
        logger.info(
            f"Num Remaining: {len(df.columns)}, Num Removed {len(cols_to_remove)}"
        )

    feat_cols = [col for col in df.columns if col not in TARGET_COLS]
    target_cols = [col for col in df.columns if col in TARGET_COLS]

    X = df[feat_cols].values
    y = df[target_cols].values

    return feat_cols, target_cols, X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OU Model Trainer")
    parser.add_argument("--config_name", type=str, default="default")
    args = parser.parse_args()

    # load config
    config_path = CONFIG_DIR / f"{args.config_name}.yaml"
    if not config_path.exists():
        raise ValueError(f"Config file: {args.config_name} does not exist")

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    benchmark_name = config['train_bench_dbs'][0]
    training_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    if config["bench_db"] not in BENCH_DBS:
        raise ValueError(f"Benchmark DB {config['bench_db']} not supported")

    benchmark_results_dir = DATA_DIR / benchmark_name

    # if no experiment name is provided, try to find one
    if config["experiment_name"] is None:
        experiment_list = sorted(
            [exp_path.name for exp_path in benchmark_results_dir.glob("*")]
        )
        logger.warning(f"{benchmark_name} experiments: {experiment_list}")
        assert len(experiment_list) > 0, f"No experiments found for {benchmark_name}"
        experiment_name = experiment_list[-1]
        logger.warning(f"Experiment name was not provided, using experiment: {experiment_name}")

    experiment_dir = benchmark_results_dir / experiment_name
    evaluation_dir = EVAL_DIR / benchmark_name / experiment_name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    ou_name_to_df, ou_name_to_nruns = load_data(experiment_dir)

    for ou_name in ou_name_to_df.keys():
        logger.warning(f"Begin Training OU: {ou_name}")
        feat_cols, target_cols, X, y = prep_data(ou_name_to_df[ou_name])
        model_name = f"{ou_name}_{training_timestamp}"

        if X.shape[1] == 0 or y.shape[1] == 0:
            logger.warning(f"{ou_name} has no valid training data, skipping")
            continue

        for method in config["methods"]:
            logger.warning(f"Training OU: {ou_name} with model: {method}")

            try:
                ou_model = BehaviorModel(method, ou_name, training_timestamp, config)
                ou_model.train(X, y)
                ou_model.save()
            except Exception as e:
                logger.warning(f"Exception encountered during training OU {ou_name}: {e}")
                continue
