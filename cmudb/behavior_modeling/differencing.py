#!/usr/bin/env python3

import os
import uuid
import pandas as pd
from config import TRAIN_DATA_ROOT, DATA_ROOT

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

    # apply differenced df back to each original dataframe in ou_to_dfs
    ou_to_diffed = {}

    for i in range(len(dfs)):
        dfs[i] = dfs[i].set_index("rid")

    for raw_df in dfs:
        raw_df.to_csv("raw_df.csv")
        diffed_df = raw_df

        ou_name = raw_df.iloc[0]["ou_name"]
        if ou_name in LEAF_NODES:
            continue

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = diffed_df.index.intersection(diffed_cols.index)

        for rid in rids_to_update:
            old_record = diffed_df.loc[rid]
            new_record = diffed_cols.loc[rid][diff_cols]

            for idx in old_record.index:
                if idx not in diff_cols:
                    new_record[idx] = old_record[idx]

            diffed_df.loc[rid] = new_record

        if "ou_name" in diffed_df.columns:
            diffed_df = diffed_df.drop(["ou_name"], axis=1)

        ou_to_diffed[ou_name] = diffed_df

    for ou_name, diffed_df in ou_to_diffed.items():
        out_path = data_path / f"{ou_name}_diffed.csv"
        diffed_df.to_csv(str(out_path), index=True)
