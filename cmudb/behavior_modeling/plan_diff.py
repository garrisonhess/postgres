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
from numpy.typing import NDArray
from pandas import DataFrame, Index
from tqdm import tqdm

from config import DATA_ROOT, LEAF_NODES, TRAIN_DATA_ROOT

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
    node_map: dict[int, list[int]] = {}
    node_map[root.plan_node_id] = [child.plan_node_id for child in root.plans]

    for child in root.plans:
        _build_id_map(child, node_map)

    return node_map


def _build_id_map(node: PlanNode, node_map: dict[int, list[int]]) -> None:
    node_map[node.plan_node_id] = [child.plan_node_id for child in node.plans]
    for child in node.plans:
        _build_id_map(child, node_map)


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


def get_plan_trees(raw_data_dir: Path, tscout_query_ids: set[str]) -> dict[str, PlanTree]:
    plan_file_path: Path = raw_data_dir / "plan_file.csv"
    cols = ["queryid", "planid", "plan"]
    dtypes: dict[str, Any] = {"queryid": np.int64, "planid": int, "plan": str}
    plan: DataFrame = pd.read_csv(plan_file_path, usecols=cols, dtype=dtypes)
    plan["queryid"] = plan["queryid"].astype(np.uint64).astype(str)

    query_id_to_plan_tree: dict[str, PlanTree] = {}

    for _, row in plan.iterrows():
        query_id: str = row["queryid"]

        if query_id in tscout_query_ids:
            json_plan: dict[str, Any] = json.loads(row["plan"])["Plan"]
            set_node_ids(json_plan)
            query_id_to_plan_tree[query_id] = PlanTree(query_id, json_plan)

    return query_id_to_plan_tree


def verify_invocation_ids(unified: DataFrame) -> None:
    inv_to_query_id: dict[int, str] = {}
    inv_to_node_ids: dict[int, set[int]] = {}

    df: DataFrame = unified[["query_id", "global_invocation_id", "plan_node_id"]].values.tolist()
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


def load_tscout_data(tscout_data_dir: Path) -> tuple[list[DataFrame], DataFrame]:
    ou_to_df: dict[str, DataFrame] = {
        f.stem: pd.read_csv(f)
        for f in tscout_data_dir.glob("*.csv")
        if f.name.startswith("Exec") and os.stat(f).st_size > 0
    }
    tscout_dfs: list[DataFrame] = []

    for ou_name, df in ou_to_df.items():
        mapper: dict[str, str] = {}
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

    unified: DataFrame = pd.concat([tdf[common_schema] for tdf in tscout_dfs], axis=0)
    unified = unified.sort_values(by=["query_id", "start_time", "plan_node_id"], axis=0)
    diff_data_dir: Path = tscout_data_dir.parent / "differenced"
    unified = add_invocation_ids(diff_data_dir, unified)
    unified, tscout_dfs = filter_incomplete(diff_data_dir, unified, tscout_dfs)

    # we use a few different indexes for unified, starting with query_id
    unified.set_index("query_id", drop=False, inplace=True)

    # we only need RID for the tscout_df index
    tscout_dfs = [df.set_index("rid", drop=False, inplace=False) for df in tscout_dfs]

    for df in tscout_dfs:
        ou_name = df.iloc[0]["ou_name"]
        df.to_csv(f"{tscout_data_dir.parent}/differenced/LOG_{ou_name}_filtered.csv")

    unified.to_csv(f"{tscout_data_dir.parent}/differenced/LOG_init_unified.csv", index=False)

    return tscout_dfs, unified


def filter_incomplete(
    diff_data_dir: Path, unified: DataFrame, tscout_dfs: list[DataFrame]
) -> tuple[DataFrame, list[DataFrame]]:
    query_id_to_node_ids: defaultdict[str, set[int]] = defaultdict(set)
    inv_id_to_node_ids: defaultdict[tuple[str, int], set[int]] = defaultdict(set)

    for (_, row) in unified.iterrows():
        query_id: str = row["query_id"]
        inv_id: int = row["global_invocation_id"]
        node_id: int = row["plan_node_id"]

        query_id_to_node_ids[query_id].add(node_id)
        inv_id_to_node_ids[(query_id, inv_id)].add(node_id)

    broken_inv_ids: set[int] = set()

    for query_id, expected_ids in query_id_to_node_ids.items():
        matched_inv_ids: list[int] = [inv_id for (q_id2, inv_id) in inv_id_to_node_ids.keys() if q_id2 == query_id]

        for inv_id in matched_inv_ids:
            actual_ids: set[int] = inv_id_to_node_ids[(query_id, inv_id)]
            symdiff: set[int] = expected_ids.symmetric_difference(actual_ids)

            if len(symdiff) > 0:
                broken_inv_ids.add(inv_id)

    unified.set_index("global_invocation_id", drop=False, inplace=True)
    working_ids: Index = unified.index.difference(broken_inv_ids)
    unified.loc[broken_inv_ids].to_csv(f"{diff_data_dir}/LOG_broken_phase2.csv")
    filt_unified: DataFrame = unified.loc[working_ids]
    filt_unified.set_index("rid", drop=False, inplace=True)

    # apply filtering to all tscout dataframes
    rid_idx: Index = Index(data=filt_unified["rid"], dtype=str)
    filt_tscout_dfs: list[DataFrame] = [filter_by_rid(rid_idx, df) for df in tscout_dfs]

    return filt_unified, filt_tscout_dfs


def filter_by_rid(rid_idx: Index, df: DataFrame) -> DataFrame:
    df.set_index("rid", drop=False, inplace=True)
    filtered_idx = rid_idx.intersection(Index(data=df["rid"], dtype=str))
    return df.loc[filtered_idx]


def add_invocation_ids(diff_data_dir: Path, unified: DataFrame) -> DataFrame:
    unified.set_index("rid", drop=False, inplace=True)

    prev_query_id: int = 0
    root_end: int = 0
    query_invocation_id: int = 0
    global_invocation_id: int = 0
    query_invocation_ids: list[int] = []
    global_invocation_ids: list[int] = []
    broken_rids: list[str] = []
    inv_cols: list[str] = ["rid", "query_id", "plan_node_id", "start_time", "end_time"]
    invocation_data: list[list[Any]] = unified[inv_cols].values.tolist()

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

    assert len(query_invocation_ids) == len(unified.index)
    working_rids: Index = unified.index.difference(broken_rids)
    unified["query_invocation_id"] = query_invocation_ids
    unified["global_invocation_id"] = global_invocation_ids

    unified.to_csv(f"{diff_data_dir}/LOG_unified_before_filtering.csv", index=False)
    unified.loc[broken_rids].to_csv(f"{diff_data_dir}/LOG_broken_phase1.csv", index=False)

    unified = unified.loc[working_rids]
    verify_invocation_ids(unified)

    return unified


def diff_one_plan(plan_tree: PlanTree, invocation: DataFrame) -> dict[str, NDArray[np.float64]]:
    rid_to_diffed_costs: dict[str, NDArray[np.float64]] = {}
    invocation.set_index("plan_node_id", drop=False, inplace=True)

    for parent_id, parent_row in invocation.iterrows():
        parent_rid: str = parent_row["rid"]
        child_ids: list[int] = plan_tree.parent_id_to_child_ids[parent_id]
        diffed_costs: NDArray[np.float64] = parent_row[diff_cols].values

        for child_id in child_ids:
            child_costs: NDArray[np.float64] = invocation.loc[child_id][diff_cols].values
            diffed_costs -= child_costs

        rid_to_diffed_costs[parent_rid] = diffed_costs

    return rid_to_diffed_costs


def save_results(diff_data_dir: Path, tscout_dfs: list[DataFrame], diffed_cols: DataFrame) -> None:

    diffed_cols.rename(columns=lambda col: f"diffed_{col}", inplace=True)

    # add the new columns onto the tscout dataframes
    for df in tscout_dfs:
        ou_name = df.iloc[0]["ou_name"]
        df.drop(["ou_name", "rid"], axis=1, inplace=True)
        if ou_name in LEAF_NODES:
            df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
            continue

        # find the intersection of RIDs between diffed_cols and each df
        rids_to_update = df.index.intersection(diffed_cols.index)
        print(f"num records to update: {rids_to_update.shape[0]}")

        if rids_to_update.shape[0] > 0:
            diffed_df = df.join(diffed_cols.loc[rids_to_update], how="inner")
            diffed_df.to_csv(f"{diff_data_dir}/{ou_name}.csv", index=True)
        else:
            diffed_df = df


def diff_all_plans(diff_data_dir: Path, unified: DataFrame) -> DataFrame:

    all_query_ids: set[str] = set(pd.unique(unified["query_id"]))
    query_id_to_plan_tree: dict[str, PlanTree] = get_plan_trees(diff_data_dir.parent, all_query_ids)
    records: list[list[Any]] = []

    for query_id in tqdm(all_query_ids):
        plan_tree: PlanTree = query_id_to_plan_tree[query_id]

        if len(plan_tree.root.plans) > 0:
            query_invocations = unified.loc[query_id]
            if isinstance(query_invocations, pd.Series):
                continue
            assert isinstance(query_invocations, DataFrame)

            query_invocation_ids: set[int] = set(pd.unique(query_invocations["query_invocation_id"]))
            print(f"Query ID: {query_id}, Number of query invocation ids: {len(query_invocation_ids)}")
            query_invocations.set_index("query_invocation_id", drop=False, inplace=True)

            for invocation_id in query_invocation_ids:
                invocation = query_invocations.loc[invocation_id]
                if isinstance(invocation, pd.Series):
                    continue
                assert isinstance(invocation, DataFrame)

                for rid, diffed_costs in diff_one_plan(plan_tree, invocation).items():
                    assert isinstance(rid, str)
                    assert isinstance(diffed_costs, np.ndarray)
                    records.append([rid] + diffed_costs.tolist())

    diffed_cols = DataFrame(data=records, columns=["rid"] + diff_cols)
    diffed_cols.to_csv(f"{diff_data_dir}/LOG_diffed_cols.csv", index=False)
    diffed_cols.set_index("rid", drop=True, inplace=True)

    return diffed_cols


def main(experiment: str) -> None:

    for mode in ["train", "eval"]:
        root_dir: Path = DATA_ROOT / mode / experiment / "tpcc"
        tscout_data_dir = root_dir / "tscout"
        diff_data_dir: Path = root_dir / "differenced"
        if diff_data_dir.exists():
            shutil.rmtree(diff_data_dir)
        diff_data_dir.mkdir()

        tscout_dfs, unified = load_tscout_data(tscout_data_dir)
        diffed_cols: DataFrame = diff_all_plans(diff_data_dir, unified)
        save_results(diff_data_dir, tscout_dfs, diffed_cols)


if __name__ == "__main__":
    # get latest experiment and run differencing
    experiment_list: list[str] = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
    if len(experiment_list) == 0:
        raise ValueError("No experiments found")

    latest_experiment: str = experiment_list[-1]
    print(f"Differencing latest experiment: {latest_experiment}")
    main(latest_experiment)
