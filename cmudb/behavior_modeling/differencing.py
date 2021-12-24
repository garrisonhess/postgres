# !/usr/bin/env python

import os
from pathlib import Path
import uuid
import pandas as pd
from config import TRAIN_DATA_ROOT

LEAF_NODES = {"ExecIndexScan", "ExecSeqScan", "ExecIndexOnlyScan", "ExecResult"}
remap_schema = [
    "plan_startup_cost",
    "plan_total_cost",
    "plan_type",
    "plan_rows",
    "plan_width",
    "plan_node_id",
]
static_schema = [
    "query_id",
    "start_time",
    "end_time",
    "cpu_id",
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

common_schema = [
    "rid",
    "query_id",
    "ou_name",
    "plan_startup_cost",
    "plan_total_cost",
    "plan_type",
    "plan_rows",
    "plan_width",
    "plan_node_id",
    "start_time",
    "end_time",
    "elapsed_us",
    "cpu_id",
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
]

experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
assert len(experiment_list) > 0, f"No experiments found"
experiment = experiment_list[-1]
print(f"Experiment name was not provided, using experiment: {experiment}")
    
for mode in ["train", "eval"]: 
    results_path = Path.home() / f"postgres/cmudb/behavior_modeling/training_data/{mode}/"
    data_path = results_path / experiment / "tpcc"
    log_path = data_path / "pg_log.txt"
    stat_path = data_path / "stat_file.csv"

    result_files = [file for file in data_path.glob("*.csv") if file.name.startswith("Exec") and "diffed" not in file.name]

    # OU
    ou_to_df = {file.stem: pd.read_csv(file) for file in result_files if os.stat(file).st_size > 0}
    dfs = []

    for ou_name, df in ou_to_df.items():
        mapper = dict()
        for col in df.columns:
            for mapper_value in remap_schema:
                if mapper_value in col:
                    mapper[col] = mapper_value
        df = df.rename(columns=mapper)
        rids = [uuid.uuid4().hex for _ in range(df.shape[0])]
        df["rid"] = rids
        df["ou_name"] = ou_name
        dfs.append(df)

    for i in range(len(dfs)):
        dfs[i] = dfs[i][common_schema]

    df = pd.concat(dfs, axis=0)
    df = df.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    diff_cols = ["plan_startup_cost", "plan_total_cost", "elapsed_us", "cpu_cycles", "instructions", "cache_references", "cache_misses",
                "ref_cpu_cycles", "network_bytes_read", "network_bytes_written", "disk_bytes_read", "disk_bytes_written", "memory_bytes"]

    for i in range(len(df.index) - 1):
        curr_record = df.iloc[i].copy()
        curr_query_id = curr_record["query_id"]
        curr_end_time = curr_record["end_time"]
        curr_ou_name = curr_record["ou_name"]

        if curr_ou_name in LEAF_NODES:
            continue

        if curr_record["plan_node_id"] > 1:
            continue

        lookahead = 1

        while True:
            if i + lookahead >= len(df.index):
                break

            next_record = df.iloc[i + lookahead]
            if curr_ou_name  == "ExecAgg" and lookahead > 1:
                break
            if next_record["start_time"] > curr_end_time or next_record["query_id"] != curr_query_id:
                break
            
            curr_record[diff_cols] -= next_record[diff_cols]
            df.iloc[i] = curr_record
            lookahead += 1

    partial_diffed_df = df.set_index("rid")

    # apply differenced df back to each original dataframe in ou_to_dfs
    ou_to_diffed = {}

    for raw_df in dfs:
        full_diffed_df = raw_df
        ou_name = raw_df.iloc[0]["ou_name"]

        for i in range(len(full_diffed_df.index)):
            curr_record = full_diffed_df.iloc[i].copy()
            curr_rid = curr_record["rid"]

            if curr_rid in partial_diffed_df.index: 
                new_subrecord = partial_diffed_df.loc[curr_rid].copy()
                curr_record[diff_cols] = new_subrecord[diff_cols]
                full_diffed_df.iloc[i] = curr_record
        
        ou_to_diffed[ou_name] = full_diffed_df

    for ou_name, diffed_df in ou_to_diffed.items():
        out_path = data_path / f"{ou_name}_diffed.csv"
        print(out_path)
        diffed_df.to_csv(str(out_path), index=False)
