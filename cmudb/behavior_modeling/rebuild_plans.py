import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

results_path = Path.home() / "postgres/cmudb/tscout/results/tpcc-default/"
experiment = "2021-12-16_20-43-45"
round = "0"
data_path = results_path / experiment / round

result_files = [file for file in data_path.glob("*.csv")]
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
    df["ou_name"] = ou_name
    dfs.append(df)

common_schema = [
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
df["query_id"] = df["query_id"].astype(np.uint64)
df.to_csv("./test_ous.csv", index=False)

###### Sample data
temp_query_id = 2655768482901157376
subdf = df[df["query_id"] == temp_query_id]
subdf = subdf[
    [
        "query_id",
        "ou_name",
        "plan_startup_cost",
        "plan_total_cost",
        "start_time",
        "end_time",
        "elapsed_us",
        "plan_node_id",
    ]
]
subdf.to_csv("./bigger_ou_results.csv", index=False)


# max_plan_node_id = df["plan_node_id"].max()
# temp = df[df["plan_node_id"] == max_plan_node_id]
# print(temp)

# for each query_id
#   verify that the count for each plan_node_id is the same
#   suppose we have plan_node_
#
#
# for each query id
# for each root node
#


def build_trees(df):

    rolled_up = df

    # for each plan node
    rolled_up[rolled_up["plan_node_id" == 0]]

    # get the min start time

    # get the max end time

    # check plan total cost behavior

    return rolled_up
