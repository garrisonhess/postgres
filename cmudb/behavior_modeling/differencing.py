#!/usr/bin/env python3

import os
import uuid

import pandas as pd
from config import DATA_ROOT, TRAIN_DATA_ROOT

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

diff_cols = [
    "plan_startup_cost",
    "plan_total_cost",
    "elapsed_us",
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

# get latest experiment
experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
assert len(experiment_list) > 0, "No experiments found"
experiment = experiment_list[-1]
print(f"Differencing latest experiment: {experiment}")

for mode in ["train", "eval"]:
    results_path = DATA_ROOT / mode
    data_path = results_path / experiment / "tpcc"
    result_files = [f for f in data_path.glob("*.csv") if f.name.startswith("Exec") and "diffed" not in f.name]
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

    unified_dfs = []
    for i in range(len(dfs)):
        unified_dfs.append(dfs[i][common_schema])

    unified_df = pd.concat(unified_dfs, axis=0)
    unified_df = unified_df.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    total_records = len(unified_df.index)
    diffed_records = []
    unified_df.to_csv("unified_df.csv")

    for i in range(total_records):
        curr_record = unified_df.iloc[i].copy()
        curr_query_id = curr_record["query_id"]
        curr_end_time = curr_record["end_time"]
        curr_ou_name = curr_record["ou_name"]

        if curr_ou_name in LEAF_NODES:
            continue

        if curr_record["plan_node_id"] > 1:
            continue

        if i % 100 == 0:
            print(f"curr record: {i} ou_name: {curr_ou_name} and total_records: {total_records}")

        lookahead = 1

        while True:
            if i + lookahead >= len(unified_df.index):
                break

            next_record = unified_df.iloc[i + lookahead]

            if curr_ou_name == "ExecAgg" and lookahead > 1:
                break
            if next_record["start_time"] > curr_end_time or next_record["query_id"] != curr_query_id:
                break

            curr_record[diff_cols] -= next_record[diff_cols]
            lookahead += 1

        diffed_records.append(curr_record)

    diffed_cols = pd.DataFrame(diffed_records)
    diffed_cols = diffed_cols.set_index("rid")
    diffed_cols.to_csv("diffed_cols.csv")

    # prepare diffed data for integration into undiffed
    nondiff_cols = [col for col in diffed_cols.columns if col not in diff_cols]
    diffed_cols = diffed_cols.drop(nondiff_cols, axis=1)
    mapper = {col: f"diffed_{col}" for col in diffed_cols.columns}
    diffed_cols = diffed_cols.rename(columns=mapper)

    # add the new columns onto the undiffed data
    ou_to_diffed = {}

    for undiffed_df in dfs:
        undiffed_df = undiffed_df.set_index("rid")
        undiffed_df.to_csv("undiffed.csv")
        ou_name = undiffed_df.iloc[0]["ou_name"]

        if "ou_name" in undiffed_df.columns:
            undiffed_df = undiffed_df.drop(["ou_name"], axis=1)

        if ou_name in LEAF_NODES:
            continue

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = undiffed_df.index.intersection(diffed_cols.index)
        assert undiffed_df.index.difference(diffed_cols.index).shape[0] == 0
        print(f"num records to update: {rids_to_update.shape[0]}")
        diffed_df = undiffed_df.join(diffed_cols, how="inner")
        ou_to_diffed[ou_name] = diffed_df

    for ou_name, diffed_df in ou_to_diffed.items():
        out_path = data_path / f"{ou_name}_diffed.csv"
        diffed_df.to_csv(str(out_path), index=True)
