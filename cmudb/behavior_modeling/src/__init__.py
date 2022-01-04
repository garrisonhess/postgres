import logging
from logging import Logger
from pathlib import Path

PG_DIR = Path.home() / "postgres"
CMUDB_DIR = PG_DIR / "cmudb"
TSCOUT_DIR = CMUDB_DIR / "tscout"
BENCHBASE_DIR = Path.home() / "benchbase"
BENCHBASE_SNAPSHOT_DIR = BENCHBASE_DIR / "benchbase-2021-SNAPSHOT"
BENCHBASE_SNAPSHOT_PATH = BENCHBASE_DIR / "target" / "benchbase-2021-SNAPSHOT.zip"
BEHAVIOR_MODELING_DIR = CMUDB_DIR / "behavior_modeling"
PG_CONF_PATH = BEHAVIOR_MODELING_DIR / "config/datagen/postgres/postgresql.conf"
CLEANUP_SCRIPT_PATH = BEHAVIOR_MODELING_DIR / "src/datagen/cleanup.py"
SQLSMITH_DIR = Path.home() / "sqlsmith"
DATA_ROOT = Path.home() / "postgres/cmudb/behavior_modeling/data/training_data"
MODELING_DIR = Path.home() / "postgres/cmudb/behavior_modeling"
MODEL_CONFIG_DIR = MODELING_DIR / "config" / "modeling"
MODEL_DIR = MODELING_DIR / "/data/models"
TRAIN_DATA_ROOT = DATA_ROOT / "/data/train"
EVAL_DATA_ROOT = DATA_ROOT / "/data/eval"


BENCH_DBS = [
    "tpcc",
    "tpch",
    "ycsb",
    "wikipedia",
    "voter",
    "twitter",
    "tatp",
    "smallbank",
    "sibench",
    "seats",
    "resourcestresser",
    "noop",
    "epinions",
    "auctionmark",
]


BENCH_TABLES = {
    "tpcc": [
        "warehouse",
        "district",
        "customer",
        "item",
        "stock",
        "oorder",
        "history",
        "order_line",
        "new_order",
    ],
    "tatp": [
        "subscriber",
        "special_facility",
        "access_info",
        "call_forwarding",
    ],
    "tpch": [
        "region",
        "nation",
        "customer",
        "supplier",
        "part",
        "orders",
        "partsupp",
        "lineitem",
    ],
    "wikipedia": [
        "useracct",
        "watchlist",
        "ipblocks",
        "logging",
        "user_groups",
        "recentchanges",
        "page",
        "revision",
        "page_restrictions",
        "text",
    ],
    "voter": [
        "contestants",
        "votes",
        "area_code_state",
    ],
    "twitter": ["user_profiles", "tweets", "follows", "added_tweets", "followers"],
    "smallbank": ["accounts", "checking", "savings"],
    "sibench": ["sitest"],
    "seats": [
        "country",
        "airline",
        "airport",
        "customer",
        "flight",
        "airport_distance",
        "frequent_flyer",
        "reservation",
        "config_profile",
        "config_histograms",
    ],
    "resourcestresser": ["iotable", "cputable", "iotablesmallrow", "locktable"],
    "noop": ["fake"],
    "epinions": ["item", "review", "useracct", "trust", "review_rating"],
    "auctionmark": [
        "region",
        "useracct",
        "category",
        "config_profile",
        "global_attribute_group",
        "item",
        "item_comment",
        "useracct_feedback",
        "useracct_attributes",
        "item_bid",
        "useracct_watch",
        "global_attribute_value",
        "item_attribute",
        "item_image",
        "item_max_bid",
        "item_purchase",
        "useracct_item",
    ],
    "ycsb": ["usertable"],
}


OU_NAMES = [
    "ExecAgg",
    "ExecAppend",
    "ExecCteScan",
    "ExecCustomScan",
    "ExecForeignScan",
    "ExecFunctionScan",
    "ExecGather",
    "ExecGatherMerge",
    "ExecGroup",
    "ExecHashJoinImpl",
    "ExecIncrementalSort",
    "ExecIndexOnlyScan",
    "ExecIndexScan",
    "ExecLimit",
    "ExecLockRows",
    "ExecMaterial",
    "ExecMergeAppend",
    "ExecMergeJoin",
    "ExecModifyTable",
    "ExecNamedTuplestoreScan",
    "ExecNestLoop",
    "ExecProjectSet",
    "ExecRecursiveUnion",
    "ExecResult",
    "ExecSampleScan",
    "ExecSeqScan",
    "ExecSetOp",
    "ExecSort",
    "ExecSubPlan",
    "ExecSubqueryScan",
    "ExecTableFuncScan",
    "ExecTidScan",
    "ExecUnique",
    "ExecValuesScan",
    "ExecWindowAgg",
    "ExecWorkTableScan",
]

METHODS = [
    "lr",
    "huber",
    "svr",
    "kr",
    "rf",
    "gbm",
    "mlp",
    "mt_lasso",
    "lasso",
    "dt",
    "mt_elastic",
    "elastic",
]


LEAF_NODES: set[str] = {"ExecIndexScan", "ExecSeqScan", "ExecIndexOnlyScan", "ExecResult"}


BASE_TARGET_COLS = [
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

DIFF_TARGET_COLS = [
    "diffed_cpu_cycles",
    "diffed_instructions",
    "diffed_cache_references",
    "diffed_cache_misses",
    "diffed_ref_cpu_cycles",
    "diffed_network_bytes_read",
    "diffed_network_bytes_written",
    "diffed_disk_bytes_read",
    "diffed_disk_bytes_written",
    "diffed_memory_bytes",
    "diffed_elapsed_us",
]

ALL_TARGET_COLS = BASE_TARGET_COLS + DIFF_TARGET_COLS

COMMON_SCHEMA: list[str] = [
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

PLAN_SCHEMA: list[str] = [
    "plan_startup_cost",
    "plan_total_cost",
    "plan_type",
    "plan_rows",
    "plan_width",
    "plan_node_id",
]

DIFF_COLS: list[str] = [
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


def get_logger() -> Logger:

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger
