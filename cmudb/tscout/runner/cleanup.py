#!/usr/bin/env python3

import psutil
import shutil
from pathlib import Path
import argparse

# TScout Processor Names
PROCESSOR_NAMES = [
    "ExecAgg Processor",
    "ExecAppend Processor",
    "ExecCteScan Processor",
    "ExecCustomScan Processor",
    "ExecForeignScan Processor",
    "ExecFunctionScan Processor",
    "ExecGather Processor",
    "ExecGatherMerge Processor",
    "ExecGroup Processor",
    "ExecHashJoinImpl Processor",
    "ExecIncrementalSort Processor",
    "ExecIndexOnlyScan Processor",
    "ExecIndexScan Processor",
    "ExecLimit Processor",
    "ExecLockRows Processor",
    "ExecMaterial Processor",
    "ExecMergeAppend Processor",
    "ExecMergeJoin Processor",
    "ExecModifyTable Processor",
    "ExecNamedTuplestoreScan Processor",
    "ExecNestLoop Processor",
    "ExecProjectSet Processor",
    "ExecRecursiveUnion Processor",
    "ExecResult Processor",
    "ExecSampleScan Processor",
    "ExecSeqScan Processor",
    "ExecSetOp Processor",
    "ExecSort Processor",
    "ExecSubPlan Processor",
    "ExecSubqueryScan Processor",
    "ExecTableFuncScan Processor",
    "ExecTidScan Processor",
    "ExecUnique Processor",
    "ExecValuesScan Processor",
    "ExecWindowAgg Processor",
    "ExecWorkTableScan Processor",
]


def kill_tscout_and_postgres():
    print("Shutting down TScout and Postgres")
    try:
        for proc in psutil.process_iter(["name"]):
            proc_name = proc.info["name"].lower()

            if "postgres" in proc_name:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
            elif "tscout" in proc_name or any(
                [processor_name.lower() in proc_name for processor_name in PROCESSOR_NAMES]
            ):
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    pass
        print("Shutdown TScout and Postgres successfully")
    except Exception as err:
        print(f"Error shutting down TScout and Postgres: {err}")


def chown_results(username):
    # change the tscout results ownership to the user who ran the benchmark
    results_dir = f"/home/{username}/postgres/cmudb/tscout/results/"
    print(f"Changing ownership of TScout results from root to user: {username}")
    shutil.chown(results_dir, user=username)
    for file in Path(results_dir).glob("**/*"):
        shutil.chown(file, user=username)
    print("Cleanup Complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Postgres/Benchbase/TScout")
    parser.add_argument("--username", help="Username to reassign file ownership", required=False)
    args = parser.parse_args()

    kill_tscout_and_postgres()

    if args.username is not None:
        chown_results(args.username)
    else:
        print("No username provided, cannot reassign result file ownership.")
