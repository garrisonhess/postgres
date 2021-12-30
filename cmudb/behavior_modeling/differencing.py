#!/usr/bin/env python3

import json
import os
import uuid
from collections import defaultdict

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
        query_id_to_plan_tree[query_id] = plan_tree

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
    unified_df.set_index("rid", drop=False, inplace=True)
    unified_df = add_invocation_ids(unified_df)
    unified_df = filter_incomplete(unified_df)

    # remove everything not in unified dfs
    unified_df.set_index("rid", drop=False, inplace=True)
    final_rids = unified_df["rid"].index

    for i in range(len(tscout_dfs)):
        tscout_dfs[i] = tscout_dfs[i].set_index("rid", drop=False)
        new_index = final_rids.intersection(tscout_dfs[i].index)
        tscout_dfs[i] = tscout_dfs[i].loc[new_index]

    if DEBUG:
        unified_df.to_csv("unified_df.csv", index=False)

    return tscout_dfs, unified_df


def filter_incomplete(unified_df):
    query_id_to_node_ids = defaultdict(set)
    inv_id_to_node_ids = defaultdict(set)

    for (_, row) in unified_df.iterrows():
        query_id = row["query_id"]
        inv_id = row["global_invocation_id"]
        node_id = row["plan_node_id"]

        query_id_to_node_ids[query_id].add(node_id)
        inv_id_to_node_ids[(query_id, inv_id)].add(node_id)

    broken_inv_ids = set()

    for query_id, expected_ids in query_id_to_node_ids.items():
        matched_inv_ids = [inv_id for (query_id2, inv_id) in inv_id_to_node_ids.keys() if query_id2 == query_id]

        for inv_id in matched_inv_ids:
            actual_ids = inv_id_to_node_ids[(query_id, inv_id)]

            symdiff = expected_ids.symmetric_difference(actual_ids)

            if len(symdiff) > 0:
                broken_inv_ids.add(inv_id)

    unified_df.set_index("global_invocation_id", drop=False, inplace=True)
    working_ids = unified_df.index.difference(broken_inv_ids)
    unified_df.loc[broken_inv_ids].to_csv("broken_invocations.csv")
    unified_df = unified_df.loc[working_ids]

    return unified_df


def add_invocation_ids(unified_df):
    prev_query_id = 0
    query_invocation_id = 0
    global_invocation_id = 0
    query_invocation_ids = []
    global_invocation_ids = []
    broken_rids = list()
    root_end = 0
    invocation_data = unified_df[["rid", "query_id", "plan_node_id", "start_time", "end_time"]].values.tolist()

    for rid, query_id, plan_node_id, curr_start, curr_end in invocation_data:
        if query_id != prev_query_id:
            root_end = curr_end
            query_invocation_id = 0
            prev_query_id = query_id
            global_invocation_id += 1
        elif plan_node_id == 0:
            root_end = curr_end
            query_invocation_id += 1
            global_invocation_id += 1
        elif curr_start > root_end:
            root_end = curr_end
            broken_rids.append(rid)
            global_invocation_id += 1

        query_invocation_ids.append(query_invocation_id)
        global_invocation_ids.append(global_invocation_id)

    assert len(query_invocation_ids) == len(unified_df.index)
    working_rids = unified_df.index.difference(broken_rids)

    unified_df["query_invocation_id"] = query_invocation_ids
    unified_df["global_invocation_id"] = global_invocation_ids
    unified_df.to_csv("unfiltered_df.csv")

    broken_df = unified_df.loc[broken_rids]
    broken_df.to_csv("broken_df.csv")

    unified_df = unified_df.loc[working_rids]

    return unified_df


def diff_costs(plan_tree, invocation_df):

    rid_to_diffed_costs = dict()
    invocation_df = invocation_df.set_index("plan_node_id")

    for parent_id, parent_row in invocation_df.iterrows():
        parent_rid = parent_row["rid"]
        child_ids = plan_tree.parent_id_to_child_ids[parent_id]
        diffed_costs = parent_row[diff_cols].values

        for child_id in child_ids:
            child_costs = invocation_df.loc[child_id][diff_cols]
            diffed_costs -= child_costs.values

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def save_results(data_dir, tscout_dfs, diffed_cols):
    # # prepare diffed data for integration into undiffed
    # nondiff_cols = [col for col in diffed_cols.columns if col not in diff_cols]
    # diffed_cols = diffed_cols.drop(nondiff_cols, axis=1)
    # mapper = {col: f"diffed_{col}" for col in diffed_cols.columns}
    # diffed_cols = diffed_cols.rename(columns=mapper)

    # add the new columns onto the undiffed data
    for undiffed_df in tscout_dfs:
        undiffed_df = undiffed_df.set_index("rid")
        ou_name = undiffed_df.iloc[0]["ou_name"]

        if "ou_name" in undiffed_df.columns:
            undiffed_df = undiffed_df.drop(["ou_name"], axis=1)

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = undiffed_df.index.intersection(diffed_cols.index)
        # assert undiffed_df.index.difference(diffed_cols.index).shape[0] == 0
        print(f"num records to update: {rids_to_update.shape[0]}")

        if rids_to_update.shape[0] > 0:
            diffed_df = undiffed_df.join(diffed_cols, how="inner")
        else:
            diffed_df = undiffed_df

        diffed_df.to_csv(f"{data_dir}/diffed_{ou_name}.csv", index=True)


def run_differencing():
    # get latest experiment
    experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
    assert len(experiment_list) > 0, "No experiments found"
    experiment = experiment_list[-1]
    print(f"Differencing latest experiment: {experiment}")

    # for mode in ["train", "eval"]:
    for mode in ["train"]:
        data_dir = DATA_ROOT / mode / experiment / "tpcc"
        tscout_dfs, unified_df = load_tscout_data(data_dir)
        query_id_to_plan_tree = get_plan_trees(data_dir, pd.unique(unified_df["query_id"]))
        unified_df.set_index("query_id", drop=False, inplace=True)
        rid_to_diffed_costs = dict()

        for query_id in tqdm(pd.unique(unified_df["query_id"])):
            plan_tree = query_id_to_plan_tree[query_id]

            if len(plan_tree.root.plans) > 0:
                query_invocations = unified_df.loc[query_id]
                query_invocation_ids = set(pd.unique(query_invocations["query_invocation_id"]))
                print(f"Query ID: {query_id}, Number of query invocation ids: {len(query_invocation_ids)}")
                query_invocations.set_index("query_invocation_id", drop=False, inplace=True)

                for invocation_id in query_invocation_ids:
                    invocation_df = query_invocations.loc[invocation_id]
                    rid_to_diffed_costs = diff_costs(plan_tree, invocation_df)
                    for k, v in rid_to_diffed_costs.items():
                        rid_to_diffed_costs[k] = v

        records = []
        for rid, diffed_costs in rid_to_diffed_costs.items():
            record = [rid] + diffed_costs.tolist()
            records.append(record)

        diffed_cols = pd.DataFrame(records, columns=["rid"] + diff_cols)
        diffed_cols.set_index("rid", drop=False, inplace=True)
        print(diffed_cols.head(1))

        save_results(data_dir, tscout_dfs, diffed_cols)


if __name__ == "__main__":
    run_differencing()
