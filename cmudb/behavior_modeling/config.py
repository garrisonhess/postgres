from pathlib import Path
import logging

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
    "hyadapt",
    "epinions",
    "chbenchmark",
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

TARGET_COLS = [
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

DATA_ROOT = Path.home() / "postgres/cmudb/behavior_modeling/training_data"
MODELING_DIR = Path.home() / "postgres/cmudb/behavior_modeling"
MODEL_CONFIG_DIR = MODELING_DIR / "config"
MODEL_DIR = MODELING_DIR / "models"
TRAIN_DATA_ROOT = DATA_ROOT / "train"
EVAL_DATA_ROOT = DATA_ROOT / "eval"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

