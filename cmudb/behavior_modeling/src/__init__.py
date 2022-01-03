import logging
from pathlib import Path

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
LEAF_NODES: set[str] = {"ExecIndexScan", "ExecSeqScan", "ExecIndexOnlyScan", "ExecResult"}


def get_logger():

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger
