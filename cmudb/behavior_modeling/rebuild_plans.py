# !/usr/bin/env python

import os
from collections import defaultdict, Counter
from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd

results_path = Path.home() / "postgres/cmudb/behavior_modeling/training_data/train/"
experiment = "experiment-2021-12-23_21-18-52"
data_path = results_path / experiment / "tpcc"
log_path = data_path / "pg_log.txt"
stat_path = data_path / "stat_file.csv"

result_files = [file for file in data_path.glob("*.csv") if file.name.startswith("Exec")]
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


for i in range(len(dfs)):
    dfs[i] = dfs[i][common_schema]

df = pd.concat(dfs, axis=0)
df = df.sort_values(by=["query_id", "start_time", "plan_total_cost"], axis=0)

df = pd.read_csv("temp.csv")
df.to_csv("undiff.csv", index=False)

diff_cols = ["plan_startup_cost", "plan_total_cost", "elapsed_us", "cpu_cycles", "instructions", "cache_references", "cache_misses",
             "ref_cpu_cycles", "network_bytes_read", "network_bytes_written", "disk_bytes_read", "disk_bytes_written", "memory_bytes"]

for i in range(len(df.index) - 1):
    curr_record = df.iloc[i].copy()
    curr_query_id = curr_record["query_id"]
    curr_end_time = curr_record["end_time"]

    # print(f"i: {i} {curr_record}")
    if curr_record["plan_node_id"] > 1:
        continue

    lookahead = 1

    while True:
        if i + lookahead >= len(df.index):
            # print(f"idx: {i + lookahead} doesn't match previous")
            break

        next_record = df.iloc[i + lookahead]
        if curr_record["ou_name"] == "ExecAgg" and lookahead > 1:
            break
        if next_record["start_time"] > curr_end_time or next_record["query_id"] != curr_query_id:
            break
        
        # print(f"idx: {i + lookahead} matches previous")

        curr_record[diff_cols] -= next_record[diff_cols]
        df.iloc[i] = curr_record
        # print(f"NEW RECORD: {curr_record}")
        lookahead += 1


df.to_csv("differenced.csv", index=False)
partial_diffed_df = df.set_index("rid")


# apply differenced df back to each original dataframe in ou_to_dfs
ou_to_diffed = {}

for raw_df in dfs:
    full_diffed_df = raw_df
    ou_name = raw_df.iloc[0]["ou_name"]
    #.set_index("rid")

    # print(full_diffed_df.columns)
    for i in range(len(full_diffed_df.index)):
        curr_record = full_diffed_df.iloc[i].copy()
        curr_rid = curr_record["rid"]

        if curr_rid in partial_diffed_df.index: 
            curr_record[diff_cols] = partial_diffed_df[curr_rid][diff_cols]
            full_diffed_df.iloc[i] = curr_record
    
    ou_to_diffed[ou_name] = full_diffed_df



for ou_name, diffed_df in ou_to_diffed.items():
    diffed_df.to_csv(f"{ou_name}_diffed.csv", index=False)



