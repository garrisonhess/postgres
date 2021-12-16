#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import numpy as np

results_path = Path.home() / "postgres/cmudb/tscout/results/tpcc-default/"
experiment = "2021-12-16_06-15-34"
round = "0"
data_path = results_path / experiment / round

result_files = [file for file in data_path.glob("*.csv")]
remap_schema = ["plan_startup_cost", "plan_total_cost", "plan_type", "plan_rows", "plan_width", "plan_node_id"]
static_schema = ["query_id",
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
                 "elapsed_us"]

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
    df["ou_name"] = ou_name
    dfs.append(df)

common_schema = ["query_id",
                 "ou_name",
                 "plan_startup_cost",
                 "plan_total_cost",
                 "plan_type",
                 "plan_rows",
                 "plan_width",
                 "plan_node_id",
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
                 "elapsed_us"]


for i in range(len(dfs)):
    dfs[i] = dfs[i][common_schema]

df = pd.concat(dfs, axis=0)
df = df.sort_values(by=["query_id", "start_time", "plan_total_cost"], axis=0)
df["query_id"] = df["query_id"].astype(np.uint64)
# df.to_csv("./test_ous.csv", index=False)

# 15237359993683966,
# 13274073564061327360
temp = df[df["plan_node_id"] == 3]
temp_query_id = df["query_id"].unique()[0]
subdf = df[df["query_id"] == temp_query_id]

print(subdf[["query_id", "ou_name", "plan_total_cost", "start_time", "end_time", "plan_node_id"]])
# print(subdf[subdf["ou_name"] == "ExecLimit"].count())
# print(subdf[subdf["ou_name"] == "ExecIndexScan"].count())
# print(subdf.count())
subdf.to_csv("./some_query.csv", index=False)





