#!/usr/bin/env python3

import psutil
import shutil
from pathlib import Path
import sys

LISTENER_NAMES = [
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


def shutdown():
    # Shutdown postgres, tscout, and benchbase
    print("Shutting down PG process and closing logfile")
    for proc in psutil.process_iter(["pid", "name", "username"]):
        if proc.info["name"].lower().startswith("postgres"):
            print(f"Killing Postgres process: {proc}")
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass
        else:
            for listener in LISTENER_NAMES:
                if listener in proc.info["name"]:
                    try:
                        print(f"Killing TScout process: {proc.info['name']}")
                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.ZombieProcess):
                        pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("USAGE: cleanup_run.py username")
        exit()
    username = sys.argv[1]

    print("Shutting down TScout and Postgres")
    shutdown()
    print("Shutdown TScout and Postgres successfully")

    # change the tscout results ownership to the user who ran the benchmark
    results_dir = f"/home/{username}/postgres/cmudb/tscout/results/"
    print(f"Changing ownership of TScout results from root to user: {username}")
    result_files = shutil.chown(results_dir, user=username)
    for file in Path(results_dir).glob("**/*"):
        shutil.chown(file, user=username)
    print("Cleanup Complete")

