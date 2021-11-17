#!/usr/bin/env python3

from subprocess import Popen
import time
import os
import argparse
from pathlib import Path
import shutil
from datetime import datetime
import psutil

BENCHMARK_NAMES = [
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


BENCHMARK_TABLES = {
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


def build_postgres(pg_dir, runner_dir):
    """Build Postgres (and extensions)"""

    try:
        os.chdir(pg_dir)
        Popen(args=["./cmudb/build/configure.sh release"], shell=True).wait()
        Popen(["make clean -s"], shell=True).wait()
        Popen(args=["make -j world-bin -s"], shell=True).wait()
        Popen(args=["make install-world-bin -j -s"], shell=True).wait()
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error building postgres")


def check_orphans():
    """Check for TScout and Postgres processes from prior runs, as they cause the runner to fail.

    This will throw an error if it finds *any* postgres processes.
    """

    tscout_process_names = ["TScout Coordinator", "TScout Processor", "TScout Collector"]
    pg_procs = []
    tscout_procs = []

    for proc in psutil.process_iter(["pid", "name", "username", "ppid", "create_time"]):
        proc_name = proc.info["name"].lower()
        if "postgres" in proc_name:
            pg_procs.append(proc)

        if any([tscout_process_name in proc.info["name"] for tscout_process_name in tscout_process_names]):
            tscout_procs.append(proc)

    assert len(pg_procs) == 0, f"Found active postgres processes from previous runs: {pg_procs}"
    assert len(tscout_procs) == 0, f"Found active tscout processes from previous runs: {tscout_procs}"


def init_pg(pg_dir, results_dir, runner_dir):
    """Initialize Postgres"""

    try:
        os.chdir(pg_dir)
        pg_log_file = open(results_dir / "pg_log.txt", "w")

        # initialize postgres for benchbase execution
        shutil.rmtree("data")
        Path("data").mkdir(parents=True, exist_ok=True)
        Popen(args=["./build/bin/initdb -D data"], shell=True).wait()

        Popen(
            args=["./build/bin/postgres -D data -W 2 &"],
            shell=True,
            stdout=pg_log_file,
            stderr=pg_log_file,
        )
        time.sleep(2)  # allows for Postgres to start up before the createdb call

        # TODO: change psql commands to use psycopg2
        Popen(args=["./build/bin/createdb test"], shell=True).wait()
        Popen(
            args=[
                '''./build/bin/psql -d test -c "CREATE ROLE admin WITH PASSWORD 'password' SUPERUSER CREATEDB CREATEROLE INHERIT LOGIN;"'''
            ],
            shell=True,
        ).wait()
        Popen(args=["./build/bin/createdb -O admin benchbase"], shell=True).wait()
        Popen(
            args=['''./build/bin/psql -d test -c "ALTER DATABASE test SET compute_query_id = 'ON';"'''],
            shell=True,
        ).wait()
        Popen(
            args=['''./build/bin/psql -d benchbase -c "ALTER DATABASE benchbase SET compute_query_id = 'ON';"'''],
            shell=True,
        ).wait()

        # Turn off pager
        Popen(
            args=['''./build/bin/psql -d benchbase -P pager=off -c "SELECT 1;"'''],
            shell=True,
        ).wait()

    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error initializing Postgres")

    return pg_log_file


def pg_prewarm(pg_dir, benchmark_name, runner_dir):
    """Prewarm Postgres so the buffer pool and OS page cache has the workload data available"""

    try:
        os.chdir(pg_dir)
        Popen(args=['''./build/bin/psql -d benchbase -c "CREATE EXTENSION pg_prewarm"'''], shell=True).wait()

        if benchmark_name not in BENCHMARK_TABLES.keys():
            raise Exception(f"Benchmark {benchmark_name} doesn't have prewarm tables setup yet.")

        for table in BENCHMARK_TABLES[benchmark_name]:
            print(f"Prewarming table: {table}")
            Popen(
                args=[f"""./build/bin/psql -d benchbase -c "select * from pg_prewarm('{table}')";"""],
                shell=True,
            ).wait()
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error prewarming Postgres")


def init_tscout(tscout_dir, results_dir, runner_dir):
    try:
        os.chdir(tscout_dir)
        Popen(
            args=[f"sudo python3 tscout.py `pgrep -ox postgres` --outdir {results_dir} &"],
            shell=True,
        )
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error initializing TScout")

    time.sleep(1)  # allows tscout to attach before Benchbase execution begins


def build_benchbase(benchbase_dir, runner_dir):
    print(f"Building Benchbase")

    try:
        os.chdir(benchbase_dir)
        benchbase_snapshot_path = benchbase_dir / "target" / "benchbase-2021-SNAPSHOT.zip"
        benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"
        Popen(args=["./mvnw clean package"], shell=True).wait()

        # TODO: resolve this in a cleaner way
        if not os.path.exists(benchbase_snapshot_dir):
            Popen(args=[f"unzip {benchbase_snapshot_path}"], shell=True).wait()
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error building benchbase")


def init_benchbase(benchbase_dir, benchmark_name, input_cfg_path, benchbase_results_dir, runner_dir):
    """Initialize Benchbase and load benchmark data"""

    try:
        os.chdir(benchbase_dir)
        benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"
        if not os.path.exists(benchbase_snapshot_dir):
            build_benchbase(benchbase_dir)
        os.chdir(benchbase_snapshot_dir)

        # move runner config to benchbase and also save it in the output directory
        benchbase_cfg_path = benchbase_snapshot_dir / f"config/postgres/{benchmark_name}_config.xml"
        shutil.copy(input_cfg_path, benchbase_cfg_path)
        shutil.copy(input_cfg_path, benchbase_results_dir)

        print(f"Initializing Benchbase for Benchmark: {benchmark_name}")
        benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=true --load=true --execute=false"
        bbase_proc = Popen(args=[benchbase_cmd], shell=True)
        bbase_proc.wait()
        if bbase_proc.returncode != 0:
            raise RuntimeError(f"Benchbase failed with return code: {bbase_proc.returncode}")
        print(f"Initialized Benchbase for Benchmark: {benchmark_name}")
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error initializing Benchbase")


def exec_benchbase(benchbase_dir, benchmark_name, benchbase_results_dir, runner_dir):
    """Execute Benchbase"""

    try:
        os.chdir(benchbase_dir)
        benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"
        if not os.path.exists(benchbase_snapshot_dir):
            build_benchbase(benchbase_dir)
        os.chdir(benchbase_snapshot_dir)

        # move runner config to benchbase and also save it in the output directory
        benchbase_cfg_path = benchbase_snapshot_dir / f"config/postgres/{benchmark_name}_config.xml"
        if not benchbase_cfg_path.exists():
            raise Exception(
                f"Benchbase config file not found. Must be setup during init_benchbase. File: {benchbase_cfg_path}"
            )

        benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=false --load=false --execute=true"
        bbase_proc = Popen(args=[benchbase_cmd], shell=True)
        bbase_proc.wait()
        if bbase_proc.returncode != 0:
            raise RuntimeError(f"Benchbase failed with return code: {bbase_proc.returncode}")

        # Move benchbase results to experiment results directory
        benchbase_stats_dir = benchbase_snapshot_dir / "results"
        # python3.9 allows for removing the str conversion
        shutil.move(str(benchbase_stats_dir), str(benchbase_results_dir))
        time.sleep(5)  # Allow TScout Collector to finish getting results
    except Exception as err:
        cleanup(runner_dir, err, terminate=True, message="Error running Benchbase")


def cleanup(runner_dir, err, terminate, message=""):
    """Clean up the TScout and Postgres processes after either a successful or failed run"""

    if len(message) > 0:
        print(message)

    if err is not None:
        print(f"Error: {err}")

    cleanup_script_path = runner_dir / "cleanup.py"
    username = psutil.Process().username()
    Popen(args=[f"sudo python3 {cleanup_script_path} --username {username}"], shell=True).wait()
    time.sleep(2)

    # Exit the program if the caller requested it (only happens on error)
    if terminate:
        exit(1)


def run(build_pg, build_bbase, benchmark_name, experiment_name, nruns, prewarm):
    """Run an experiment"""

    pg_dir = Path.home() / "postgres"
    cmudb_dir = pg_dir / "cmudb"
    tscout_dir = cmudb_dir / "tscout"
    runner_dir = tscout_dir / "runner"
    benchbase_dir = Path.home() / "benchbase"
    benchmark_cfg_path = runner_dir / "benchbase_configs" / f"{benchmark_name}_config.xml"
    experiment_dir = tscout_dir / "results" / benchmark_name / experiment_name
    Path(experiment_dir).mkdir(parents=True, exist_ok=True)

    if build_pg:
        build_postgres(pg_dir, runner_dir)

    if build_bbase:
        build_benchbase(benchbase_dir, runner_dir)

    print(f"Running experiment: {experiment_name} with {nruns} runs and experiment output dir: {experiment_dir}")

    for run_id in range(nruns):
        print(f"Starting run {run_id}")
        check_orphans()  # Check for orphaned processes from prior runs

        # Create output directories for this run
        results_dir = experiment_dir / str(run_id)
        Path(results_dir).mkdir(exist_ok=True)
        benchbase_results_dir = results_dir / "benchbase"
        Path(benchbase_results_dir).mkdir(exist_ok=True)

        pg_log_file = init_pg(pg_dir, results_dir, runner_dir)
        init_benchbase(benchbase_dir, benchmark_name, benchmark_cfg_path, benchbase_results_dir, runner_dir)

        if prewarm:
            pg_prewarm(pg_dir, benchmark_name, runner_dir)

        init_tscout(tscout_dir, results_dir, runner_dir)
        exec_benchbase(benchbase_dir, benchmark_name, benchbase_results_dir, runner_dir)

        pg_log_file.close()
        cleanup(runner_dir, err=None, terminate=False, message=f"Finished run {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with Postgres, Benchbase, and TScout")
    parser.add_argument("--build-pg", action="store_true", default=False)
    parser.add_argument("--build-bbase", action="store_true", default=False)
    parser.add_argument("--benchmark-name", default="tpcc")
    parser.add_argument("--experiment-name", default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    parser.add_argument("--nruns", type=int, default=1)
    parser.add_argument("--no-prewarm", action="store_true", default=False)

    args = parser.parse_args()
    build_pg = args.build_pg
    build_bbase = args.build_bbase
    benchmark_name = args.benchmark_name
    experiment_name = args.experiment_name
    nruns = args.nruns
    prewarm = not args.no_prewarm

    if nruns <= 0 or nruns > 10:
        raise ValueError("Invalid nruns: {runs}")

    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

    run(build_pg, build_bbase, benchmark_name, experiment_name, nruns, prewarm)
