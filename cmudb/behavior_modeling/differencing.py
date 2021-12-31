#!/usr/bin/env python3

import json
import os
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Index
from tqdm import tqdm

from config import DATA_ROOT, TRAIN_DATA_ROOT

LEAF_NODES: set[str] = {"ExecIndexScan", "ExecSeqScan", "ExecIndexOnlyScan", "ExecResult"}

remap_schema: list[str] = [
    "plan_startup_cost",
    "plan_total_cost",
    "plan_type",
    "plan_rows",
    "plan_width",
    "plan_node_id",
]

static_schema: list[str] = [
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

common_schema: list[str] = [
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

diff_cols: list[str] = [
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

DEBUG: bool = True


class PlanTree:
    def __init__(self, query_id: str, json_plan: dict[str, Any]):
        self.query_id: str = query_id
        self.root: PlanNode = PlanNode(json_plan)
        self.parent_id_to_child_ids: dict[int, list[int]] = build_id_map(self.root)


class PlanNode:
    def __init__(self, json_plan: dict[str, Any]):
        self.node_type: str = json_plan["Node Type"]
        self.startup_cost: float = json_plan["Startup Cost"]
        self.total_cost: float = json_plan["Total Cost"]
        self.plan_node_id: int = json_plan["plan_node_id"]
        self.depth: int = json_plan["depth"]
        self.plans: list[PlanNode] = (
            [PlanNode(child_plan) for child_plan in json_plan["Plans"]] if "Plans" in json_plan else []
        )

    def __repr__(self) -> str:
        indent = ">" * (self.depth + 1)  # incremented so we always prefix with >
        return f"{indent} type: {self.node_type}, total_cost: {self.total_cost}, node_id: {self.plan_node_id}, depth: {self.depth}"


def build_id_map(root: PlanNode) -> dict[int, list[int]]:
    map: dict[int, list[int]] = dict()
    map[root.plan_node_id] = [child.plan_node_id for child in root.plans]

    for child in root.plans:
        _build_id_map(child, map)

    return map


def _build_id_map(node: PlanNode, map: dict[int, list[int]]) -> None:
    map[node.plan_node_id] = [child.plan_node_id for child in node.plans]
    for child in node.plans:
        _build_id_map(child, map)


def show_plan_tree(plan_tree: PlanTree) -> None:
    print(f"\n===== QueryID: {plan_tree.query_id} =====")
    print(f"Parent ID to Child IDs: {plan_tree.parent_id_to_child_ids}")
    print(plan_tree.root)

    for child in plan_tree.root.plans:
        show_plan_node(child)


def show_plan_node(plan_node: PlanNode) -> None:
    print(plan_node)

    for child in plan_node.plans:
        show_plan_node(child)


def set_node_ids(json_plan: dict[str, Any]) -> None:
    json_plan["plan_node_id"] = 0
    json_plan["depth"] = 0
    next_node_id: int = 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, 1)


def _set_node_ids(json_plan: dict[str, Any], next_node_id: int, depth: int) -> int:
    json_plan["plan_node_id"] = next_node_id
    json_plan["depth"] = depth
    next_node_id += 1

    if "Plans" in json_plan:
        for child in json_plan["Plans"]:
            next_node_id = _set_node_ids(child, next_node_id, depth + 1)

    return next_node_id


# TODO: figure out query_id type
def get_plan_trees(data_dir: Path, tscout_query_ids: set[str]) -> dict[str, PlanTree]:
    plan_file_path: Path = data_dir / "plan_file.csv"
    cols = ["queryid", "planid", "plan"]
    dtypes: dict[str, Any] = {"queryid": np.int64, "planid": int, "plan": str}
    plan_df: DataFrame = pd.read_csv(plan_file_path, usecols=cols, dtype=dtypes)
    plan_df["queryid"] = plan_df["queryid"].astype(np.uint64).astype(str)

    query_id_to_plan_tree: dict[str, PlanTree] = dict()

    for i in range(len(plan_df.index)):

        query_id = plan_df.iloc[i]["queryid"]

        if query_id in tscout_query_ids:
            plan: str = plan_df.iloc[i]["plan"]
            json_plan: dict[str, Any] = json.loads(plan)
            json_plan = json_plan["Plan"]
            set_node_ids(json_plan)
            query_id_to_plan_tree[query_id] = PlanTree(query_id, json_plan)

    return query_id_to_plan_tree


def load_tscout_data(data_dir: Path) -> tuple[list[DataFrame], DataFrame]:
    result_files: list[Path] = [f for f in data_dir.glob("*.csv") if f.name.startswith("Exec")]
    ou_to_df: dict[str, DataFrame] = {
        file.stem: pd.read_csv(file) for file in result_files if os.stat(file).st_size > 0
    }
    tscout_dfs: list[DataFrame] = []

    for ou_name, df in ou_to_df.items():
        mapper: dict[str, str] = dict()
        for col in df.columns:
            for mapper_value in remap_schema:
                if mapper_value in col:
                    mapper[col] = mapper_value
        df = df.rename(columns=mapper)
        rids: list[str] = [uuid.uuid4().hex for _ in range(df.shape[0])]
        df["rid"] = rids
        df["ou_name"] = ou_name
        df["query_id"] = df["query_id"].astype(str)
        tscout_dfs.append(df)

    unified_dfs: list[DataFrame] = []
    for i in range(len(tscout_dfs)):
        unified_dfs.append(tscout_dfs[i][common_schema])

    unified_df: DataFrame = pd.concat(unified_dfs, axis=0)
    unified_df = unified_df.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    unified_df.set_index("rid", drop=False, inplace=True)
    unified_df = add_invocation_ids(data_dir, unified_df)
    unified_df = filter_incomplete(data_dir, unified_df)

    # remove everything not in unified dfs
    unified_df.set_index("rid", drop=False, inplace=True)
    final_rids: list[str] = unified_df["rid"].values.tolist()
    rid_index: Index[str] = Index(data=final_rids, dtype=str)

    for i in range(len(tscout_dfs)):
        tscout_dfs[i] = tscout_dfs[i].set_index("rid", drop=False)
        new_index: Index[int] = rid_index.intersection(tscout_dfs[i].index)
        tscout_dfs[i] = tscout_dfs[i].loc[new_index]

    if DEBUG:
        unified_df.to_csv(f"{data_dir}/differencing/init_unified_df.csv", index=False)

    return tscout_dfs, unified_df


def filter_incomplete(data_dir: Path, unified_df: DataFrame) -> DataFrame:
    query_id_to_node_ids: defaultdict[str, set[int]] = defaultdict(set)
    inv_id_to_node_ids: defaultdict[tuple[str, int], set[int]] = defaultdict(set)

    for (_, row) in unified_df.iterrows():
        query_id: str = row["query_id"]
        inv_id: int = row["global_invocation_id"]
        node_id: int = row["plan_node_id"]

        query_id_to_node_ids[query_id].add(node_id)
        inv_id_to_node_ids[(query_id, inv_id)].add(node_id)

    broken_inv_ids: set[int] = set()

    for query_id, expected_ids in query_id_to_node_ids.items():
        matched_inv_ids: list[int] = [
            inv_id for (query_id2, inv_id) in inv_id_to_node_ids.keys() if query_id2 == query_id
        ]

        for inv_id in matched_inv_ids:
            actual_ids: set[int] = inv_id_to_node_ids[(query_id, inv_id)]
            symdiff: set[int] = expected_ids.symmetric_difference(actual_ids)

            if len(symdiff) > 0:
                broken_inv_ids.add(inv_id)

    unified_df.set_index("global_invocation_id", drop=False, inplace=True)
    working_ids: Index[str] = unified_df.index.difference(broken_inv_ids)

    if DEBUG:
        broken_invs: DataFrame = unified_df.loc[broken_inv_ids]
        broken_invs.to_csv(f"{data_dir}/differencing/broken_invocations.csv")

    unified_df: DataFrame = unified_df.loc[working_ids]

    return unified_df


def add_invocation_ids(data_dir: Path, unified_df: DataFrame) -> DataFrame:
    prev_query_id: int = 0
    root_end: int = 0
    query_invocation_id: int = 0
    global_invocation_id: int = 0
    query_invocation_ids: list[int] = []
    global_invocation_ids: list[int] = []
    broken_rids: list[int] = list()
    invocation_data: list[Any] = unified_df[
        ["rid", "query_id", "plan_node_id", "start_time", "end_time"]
    ].values.tolist()

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
    working_rids: Index[str] = unified_df.index.difference(broken_rids)
    unified_df["query_invocation_id"] = query_invocation_ids
    unified_df["global_invocation_id"] = global_invocation_ids

    if DEBUG:
        unified_df.to_csv(f"{data_dir}/differencing/unified_df_before_filtering.csv", index=False)
        broken_df: DataFrame = unified_df.loc[broken_rids]
        broken_df.to_csv(f"{data_dir}/differencing/broken_df.csv", index=False)

    unified_df = unified_df.loc[working_rids]
    verify_ids(unified_df)

    return unified_df


def diff_costs(plan_tree: PlanTree, invocation_df: DataFrame) -> dict[str, ndarray[Any, Any]]:
    rid_to_diffed_costs: dict[str, ndarray[Any, Any]] = dict()
    invocation_df.set_index("plan_node_id", drop=False, inplace=True)

    for parent_id, parent_row in invocation_df.iterrows():
        parent_rid: str = parent_row["rid"]
        child_ids: list[int] = plan_tree.parent_id_to_child_ids[parent_id]
        diffed_costs: ndarray[Any, Any] = parent_row[diff_cols].values

        for child_id in child_ids:
            child_costs: ndarray[Any, Any] = invocation_df.loc[child_id][diff_cols].values
            diffed_costs -= child_costs

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def save_results(data_dir: Path, tscout_dfs: list[DataFrame], diffed_cols: DataFrame) -> None:
    # add the new columns onto the undiffed data
    for undiffed_df in tscout_dfs:
        undiffed_df = undiffed_df.set_index("rid")
        ou_name = undiffed_df.iloc[0]["ou_name"]

        if "ou_name" in undiffed_df.columns:
            undiffed_df = undiffed_df.drop(["ou_name"], axis=1)

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = undiffed_df.index.intersection(diffed_cols.index)
        assert undiffed_df.index.difference(diffed_cols.index).shape[0] == 0
        print(f"num records to update: {rids_to_update.shape[0]}")

        if rids_to_update.shape[0] > 0:
            diffed_df = undiffed_df.join(diffed_cols, how="inner")
        else:
            diffed_df = undiffed_df

        diffed_df.to_csv(f"{data_dir}/diffed_{ou_name}.csv", index=True)


def verify_ids(unified_df: DataFrame) -> None:
    inv_to_query_id: dict[int, str] = dict()
    inv_to_node_ids: dict[int, set[int]] = dict()

    df: DataFrame = unified_df[["query_id", "global_invocation_id", "plan_node_id"]].values.tolist()
    for query_id, inv_id, node_id in df:

        # verify each global_invocation_id maps to the same query_id
        if inv_id in inv_to_query_id:
            old_query_id = inv_to_query_id[inv_id]
            assert (
                query_id == old_query_id
            ), f"Found conflicting query_ids for inv_id: {inv_id}, new_query_id: {query_id}, old_query_id: {old_query_id}"
        else:
            inv_to_query_id[inv_id] = query_id

        # verify each global_invocation_id has no duplicate plan_node_ids
        if inv_id in inv_to_node_ids:
            assert (
                node_id not in inv_to_node_ids[inv_id]
            ), f"Found duplicate plan_node_id: {node_id} for inv_id: {inv_id}"
            inv_to_node_ids[inv_id].add(node_id)
        else:
            inv_to_node_ids[inv_id] = {node_id}

    return None


def run_differencing(experiment: str) -> None:

    # for mode in ["train", "eval"]:
    for mode in ["train"]:
        data_dir: Path = DATA_ROOT / mode / experiment / "tpcc"
        diff_dir: Path = Path(data_dir / "differencing")
        if diff_dir.exists():
            shutil.rmtree(diff_dir)
        diff_dir.mkdir(parents=True, exist_ok=True)

        tscout_dfs, unified_df = load_tscout_data(data_dir)
        all_query_ids: set[str] = set(pd.unique(unified_df["query_id"]))
        query_id_to_plan_tree: dict[str, PlanTree] = get_plan_trees(data_dir, all_query_ids)
        unified_df.set_index("query_id", drop=False, inplace=True)
        rid_to_diffed_costs: dict[str, ndarray[Any, Any]] = dict()

        query_ids: set[str] = set(unified_df["query_id"].values.tolist())

        for query_id in tqdm(query_ids):
            plan_tree: PlanTree = query_id_to_plan_tree[query_id]

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

        records: list[list[Any]] = []
        for rid, diffed_costs in rid_to_diffed_costs.items():
            record: list[Any] = [rid] + diffed_costs.tolist()
            records.append(record)

        diffed_cols: DataFrame = DataFrame(records, columns=["rid"] + diff_cols)
        diffed_cols.set_index("rid", drop=False, inplace=True)
        diffed_cols.to_csv(f"{data_dir}/differencing/diffed_cols.csv")
        print(diffed_cols.head(1))

        save_results(data_dir, tscout_dfs, diffed_cols)


if __name__ == "__main__":
    # get latest experiment and run differencing
    experiment_list: list[str] = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
    if len(experiment_list) == 0:
        raise ValueError("No experiments found")

    experiment: str = experiment_list[-1]
    print(f"Differencing latest experiment: {experiment}")
    run_differencing(experiment)
