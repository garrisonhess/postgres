#!/usr/bin/env python3

import psutil
import shutil
from pathlib import Path
import sys

LISTENER_NAMES = [
    "ExecAgg",
    "ExecAppe",
    "ExecCteS",
    "ExecCust",
    "ExecFore",
    "ExecFunc",
    "ExecGath",
    "ExecGath",
    "ExecGrou",
    "ExecHash",
    "ExecIncr",
    "ExecInde",
    "ExecInde",
    "ExecLimi",
    "ExecLock",
    "ExecMate",
    "ExecMerg",
    "ExecMerg",
    "ExecModi",
    "ExecName",
    "ExecNest",
    "ExecProj",
    "ExecRecu",
    "ExecResu",
    "ExecSamp",
    "ExecSeqS",
    "ExecSetO",
    "ExecSort",
    "ExecSubP",
    "ExecSubq",
    "ExecTabl",
    "ExecTidS",
    "ExecUniq",
    "ExecValu",
    "ExecWind",
    "ExecWork",
]


def shutdown():
    # Shutdown postgres, tscout, and benchbase
    print("Shutting down PG process and closing logfile")
    for proc in psutil.process_iter(["pid", "name", "username"]):
        proc_name = proc.info["name"].lower()
        if proc_name.startswith("postgres") or proc_name.startswith("tscout"):
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                pass

        else:
            for listener in LISTENER_NAMES:
                if proc_name.startswith(listener.lower()):
                    try:
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
