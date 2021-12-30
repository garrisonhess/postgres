#!/usr/bin/env python3

import json
import os
import uuid

import numpy as np
import pandas as pd
from config import DATA_ROOT, TRAIN_DATA_ROOT
from tqdm import tqdm

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

DEBUG = True


class PlanTree:
    def __init__(self, query_id, json_plan):
        self.query_id = query_id
        self.json_plan = json_plan
        self.root = PlanNode(json_plan)
        self.parent_id_to_child_ids = build_id_map(self.root)


def build_id_map(root):
    map = dict()
    map[root.plan_node_id] = [child.plan_node_id for child in root.plans]

    for child in root.plans:
        _build_id_map(child, map)

    return map


def _build_id_map(node, map):

    map[node.plan_node_id] = [child.plan_node_id for child in node.plans]

    for child in node.plans:
        _build_id_map(child, map)


class PlanNode:
    def __init__(self, json_plan):
        self.node_type = json_plan["Node Type"]
        self.startup_cost = json_plan["Startup Cost"]
        self.total_cost = json_plan["Total Cost"]
        self.plan_node_id = json_plan["plan_node_id"]
        self.depth = json_plan["depth"]
        self.plans = [PlanNode(child_plan) for child_plan in json_plan["Plans"]] if "Plans" in json_plan else []

    def __repr__(self):
        indent = ">" * (self.depth + 1)  # incremented so we always prefix with >
        return f"{indent} type: {self.node_type}, total_cost: {self.total_cost}, node_id: {self.plan_node_id}, depth: {self.depth}"


def show_plan_tree(plan_tree):
    print(f"\n===== QueryID: {plan_tree.query_id} =====")
    print(f"Parent ID to Child IDs: {plan_tree.parent_id_to_child_ids}")
    print(plan_tree.root)

    for child in plan_tree.root.plans:
        show_plan_node(child)


def show_plan_node(plan_node):
    print(plan_node)

    for child in plan_node.plans:
        show_plan_node(child)


def set_node_ids(json_plan):
    json_plan["plan_node_id"] = 0
    json_plan["depth"] = 0

    next_node_id = 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, 1)


def _set_node_ids(json_plan, next_node_id, depth):

    json_plan["plan_node_id"] = next_node_id
    json_plan["depth"] = depth
    next_node_id += 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, depth + 1)

    return next_node_id


# def differencing():
#     for i in range(total_records):
#         curr_record = unified_df.iloc[i].copy()
#         curr_query_id = curr_record["query_id"]
#         curr_end_time = curr_record["end_time"]
#         curr_ou_name = curr_record["ou_name"]

#         if curr_ou_name in LEAF_NODES:
#             continue

#         if curr_record["plan_node_id"] > 1:
#             continue

#         if i % 100 == 0:
#             print(f"curr record: {i} ou_name: {curr_ou_name} and total_records: {total_records}")

#         lookahead = 1

#         while True:
#             if i + lookahead >= len(unified_df.index):
#                 break

#             next_record = unified_df.iloc[i + lookahead]

#             if curr_ou_name == "ExecAgg" and lookahead > 1:
#                 break
#             if next_record["start_time"] > curr_end_time or next_record["query_id"] != curr_query_id:
#                 break

#             curr_record[diff_cols] -= next_record[diff_cols]
#             lookahead += 1

#         diffed_records.append(curr_record)

#     diffed_cols = pd.DataFrame(diffed_records)
#     diffed_cols = diffed_cols.set_index("rid")

#     if DEBUG:
#         diffed_cols.to_csv("diffed_cols.csv")

#     # prepare diffed data for integration into undiffed
#     nondiff_cols = [col for col in diffed_cols.columns if col not in diff_cols]
#     diffed_cols = diffed_cols.drop(nondiff_cols, axis=1)
#     mapper = {col: f"diffed_{col}" for col in diffed_cols.columns}
#     diffed_cols = diffed_cols.rename(columns=mapper)

#     # add the new columns onto the undiffed data
#     ou_to_diffed = {}

#     for undiffed_df in tscout_dfs:
#         undiffed_df = undiffed_df.set_index("rid")
#         ou_name = undiffed_df.iloc[0]["ou_name"]

#         if "ou_name" in undiffed_df.columns:
#             undiffed_df = undiffed_df.drop(["ou_name"], axis=1)

#         if ou_name in LEAF_NODES:
#             continue

#         # find the intersection of RIDs between diffed_cols and each df
#         rids_to_update = undiffed_df.index.intersection(diffed_cols.index)
#         assert undiffed_df.index.difference(diffed_cols.index).shape[0] == 0
#         print(f"num records to update: {rids_to_update.shape[0]}")
#         diffed_df = undiffed_df.join(diffed_cols, how="inner")
#         ou_to_diffed[ou_name] = diffed_df

#     for ou_name, diffed_df in ou_to_diffed.items():
#         out_path = data_dir / f"{ou_name}.csv"
#         diffed_df.to_csv(str(out_path), index=True)


def get_plan_trees(data_dir, tscout_query_ids):
    plan_file_path = data_dir / "plan_file.csv"
    plan_df = pd.read_csv(plan_file_path)
    query_id_to_plan = dict()
    plan_df = plan_df[["queryid", "planid", "plan"]]
    plan_df["queryid"] = plan_df["queryid"].astype(np.uint64)

    for i in range(len(plan_df.index)):
        query_id = plan_df.iloc[i]["queryid"]
        if query_id in tscout_query_ids:
            plan = plan_df.iloc[i]["plan"]
            query_id_to_plan[query_id] = json.loads(plan)

    query_id_to_plan_tree = dict()

    for query_id, json_plan in query_id_to_plan.items():
        set_node_ids(json_plan["Plan"])
        plan_tree = PlanTree(query_id, json_plan["Plan"])
        show_plan_tree(plan_tree)
        # query_id_to_plan_tree[query_id] = plan_tree

    return query_id_to_plan_tree


def load_tscout_data(data_dir):
    result_files = [f for f in data_dir.glob("*.csv") if f.name.startswith("Exec")]
    ou_to_df = {file.stem: pd.read_csv(file) for file in result_files if os.stat(file).st_size > 0}
    tscout_dfs = []

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
        df["query_id"] = df["query_id"].astype(np.uint64)
        tscout_dfs.append(df)

    unified_dfs = []
    for i in range(len(tscout_dfs)):
        unified_dfs.append(tscout_dfs[i][common_schema])

    unified_df = pd.concat(unified_dfs, axis=0)
    unified_df = unified_df.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)

    tscout_query_ids = set(pd.unique(unified_df["query_id"]))
    unified_df = add_invocation_ids(unified_df)

    if DEBUG:
        unified_df.to_csv("unified_df.csv")

    return tscout_dfs, unified_df, tscout_query_ids


def add_invocation_ids(unified_df):
    prev_query_id = 0
    query_invocation_id = 0
    global_invocation_id = 0
    query_invocation_ids = []
    global_invocation_ids = []

    for query_id, plan_node_id in unified_df[["query_id", "plan_node_id"]].values.tolist():
        if query_id != prev_query_id:
            query_invocation_id = 0
            prev_query_id = query_id
        elif plan_node_id == 0:
            query_invocation_id += 1
            global_invocation_id += 1

        query_invocation_ids.append(query_invocation_id)
        global_invocation_ids.append(global_invocation_id)

    assert len(query_invocation_ids) == len(unified_df.index)
    unified_df["query_invocation_id"] = query_invocation_ids
    unified_df["global_invocation_id"] = global_invocation_ids

    return unified_df


def diff_costs(plan_tree, invocation_df):

    diffed_rows = []

    for idx, parent_row in invocation_df.iterrows():
        parent_id = parent_row["plan_node_id"]
        child_ids = plan_tree.parent_id_to_child_ids[parent_id]

        for child_id in child_ids:
            child_row = invocation_df[invocation_df["plan_node_id"] == child_id]
            parent_row -= child_row

        diffed_rows.append(parent_row)
        print(f"index: {idx}, row: {parent_row}")

    diffed = pd.DataFrame(diffed_rows)

    return diffed


if __name__ == "__main__":
    # get latest experiment
    experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
    assert len(experiment_list) > 0, "No experiments found"
    experiment = experiment_list[-1]
    print(f"Differencing latest experiment: {experiment}")

    # for mode in ["train", "eval"]:
    for mode in ["train"]:
        data_dir = DATA_ROOT / mode / experiment / "tpcc"
        tscout_dfs, unified_df, tscout_query_ids = load_tscout_data(data_dir)
        query_id_to_plan_tree = get_plan_trees(data_dir, tscout_query_ids)
        query_invocation_ids = set(pd.unique(unified_df["query_invocation_id"]))
        unified_df.set_index("query_invocation_id")

        print(f"Number of query invocation ids: {len(query_invocation_ids)}")

        for invocation_id in query_invocation_ids:
            invocation_df = unified_df.loc[invocation_id]
            plan_tree = query_id_to_plan_tree[invocation_df.iloc[0]["query_id"]]

            updated = diff_costs(plan_tree, invocation_df)

            print(updated)
            break
        #

        print(unified_df.head(10))

        # then match every invocation to its plan_tree
        # for each global_invocation_id
        #   for each record
        #     find the corresponding plan tree node
        #     find

        #

        # we want something like RID -> ChildRIDS

        # then we can go df[rid][diff_cols] -= df[child_rids][diff_cols]
