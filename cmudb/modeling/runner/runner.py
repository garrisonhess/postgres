#!/usr/bin/env python3

from subprocess import PIPE, Popen
import time
import os
import argparse
from pathlib import Path
import shutil
from datetime import datetime
import psutil


def build_postgres(pg_dir):
    os.chdir(pg_dir)
    print("Building Postgres")
    Popen(args=["./cmudb/build/configure.sh release"], shell=True).wait()
    Popen(["make clean -s"], shell=True).wait()
    Popen(args=["make -j -s"], shell=True).wait()
    Popen(args=["make install -j -s"], shell=True).wait()
    print("Built Postgres")


# TODO: change psql commands to use psycopg2
def init_pg(pg_dir, output_dir):
    os.chdir(pg_dir)
    pg_log_file = open(output_dir / "pg_log.txt", "w")

    # initialize postgres for benchbase execution
    shutil.rmtree("data")
    Path("data").mkdir(parents=True, exist_ok=True)
    Popen(args=["./build/bin/initdb -D data"], shell=True).wait()

    print("Starting Postgres")
    pg_proc = Popen(
        args=["./build/bin/postgres -D data -W 2 &"],
        shell=True,
        stdout=pg_log_file,
        stderr=pg_log_file,
    )
    time.sleep(5)
    print("Started Postgres")

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
        args=[
            '''./build/bin/psql -d benchbase -c "ALTER DATABASE benchbase SET compute_query_id = 'ON';"'''
        ],
        shell=True,
    ).wait()

    return pg_proc, pg_log_file


def init_tscout(tscout_dir, benchmark_name, experiment_name, run_id):
    print(f"Starting TScout for {benchmark_name}, {experiment_name}, run_id: {run_id}")
    os.chdir(tscout_dir)

    pg_pid = None
    for proc in psutil.process_iter(["pid", "name"]):
        if proc["name"] == "postgres":
            if pg_pid is not None:
                raise Exception("Found multiple postgres processes")
            pg_pid = proc["pid"]

    if pg_pid is None:
        raise Exception("Failed to find postgres process")

    tscout_proc = Popen(
        args=[f"sudo python3 tscout.py {pg_pid} {benchmark_name} {experiment_name} {run_id} &"],
        shell=True,
    )
    time.sleep(10)
    return tscout_proc


def build_benchbase(benchbase_dir):
    os.chdir(benchbase_dir)

    benchbase_snapshot_path = benchbase_dir / "target" / "benchbase-2021-SNAPSHOT.zip"
    benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"

    # build benchbase and setup tpc-c
    print(f"Building Benchbase from dir: {os.getcwd()}")
    Popen(args=["./mvnw clean package"], shell=True).wait()

    if not os.path.exists(benchbase_snapshot_dir):
        Popen(args=[f"unzip {benchbase_snapshot_path}"], shell=True).wait()


def init_benchbase(
    benchbase_dir, benchmark_name, input_cfg_path,
):
    os.chdir(benchbase_dir)
    benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"
    if not os.path.exists(benchbase_snapshot_dir):
        build_benchbase(benchbase_dir)
    os.chdir(benchbase_snapshot_dir)

    # move runner config to benchbase and also save it in the output directory
    benchbase_cfg_path = benchbase_snapshot_dir / f"config/postgres/{benchmark_name}_config.xml"
    shutil.copy(input_cfg_path, benchbase_cfg_path)
    shutil.copy(input_cfg_path, output_dir)

    print(f"Initializing Benchbase for Benchmark: {benchmark_name}")
    benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=true --load=true --execute=false"
    Popen(args=[benchbase_cmd], shell=True).wait()
    print(f"Initialized Benchbase for Benchmark: {benchmark_name}")


def exec_benchbase(benchbase_dir, benchmark_name, benchbase_run_results_dir):
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

    print("Starting Benchbase")
    benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=false --load=false --execute=true"
    Popen(args=[benchbase_cmd], shell=True).wait()

    # Copy benchbase results to experiment results directory
    benchbase_results_dir = benchbase_snapshot_dir / "results"
    print(f"Copying Benchbase results from {benchbase_results_dir} to {benchbase_run_results_dir}")
    # result_files = benchbase_results_dir.glob("**/*")
    shutil.move(benchbase_results_dir, benchbase_run_results_dir)
    # for file in result_files:
    #     print(f"Copying {file} to {benchbase_run_results_dir}")
    #     shutil.copy(file, benchbase_run_results_dir)


def cleanup(runner_dir, err, message=""):
    print(message)
    print(err)
    cleanup_script_path = runner_dir / "cleanup_run.py"
    username = psutil.Process().username()
    Popen(args=[f"sudo python3 {cleanup_script_path} {username}"], shell=True).wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Postgres/Benchbase/TScout")
    parser.add_argument("--build-benchbase", dest="build_bbase", action="store_true", default=False)
    parser.add_argument("--build-pg", dest="build_pg", action="store_true", default=False)
    parser.add_argument("--benchmark-name", dest="benchmark_name", action="store", default="tpcc")
    parser.add_argument("--experiment-name", dest="experiment_name", action="store", default="")
    parser.add_argument("--nruns", dest="nruns", action="store", default=1)

    args = parser.parse_args()
    build_bbase = args.build_bbase
    benchmark_name = args.benchmark_name
    experiment_name = args.experiment_name
    build_pg = args.build_pg
    build_bbase = args.build_bbase
    nruns = args.nruns
    valid_benchmarks = ["tpcc", "tpch"]

    if len(experiment_name) == 0:
        experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if benchmark_name not in valid_benchmarks:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

    pg_dir = Path.home() / "postgres"
    cmudb_dir = pg_dir / "cmudb"
    tscout_dir = cmudb_dir / "tscout"
    modeling_dir = cmudb_dir / "modeling"
    runner_dir = modeling_dir / "runner"
    benchmark_cfg_path = runner_dir / "benchbase_configs" / f"{benchmark_name}_config.xml"
    benchbase_dir = Path.home() / "benchbase"
    output_dir = tscout_dir / "results" / benchmark_name / experiment_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    benchbase_output_dir = tscout_dir / "benchbase_results"

    if build_pg:
        try:
            build_postgres(pg_dir)
        except Exception as err:
            cleanup(runner_dir, err, message="Error building postgres")

    if build_bbase:
        try:
            build_benchbase(benchbase_dir)
        except Exception as err:
            cleanup(runner_dir, err, message="Error building benchbase")

    print(f"Running experiment: {experiment_name} with {nruns} runs and output_dir: {output_dir}")
    for run_id in range(nruns):

        print(f"Starting run {run_id}")
        try:
            pg_proc, pg_log_file = init_pg(pg_dir, output_dir)
        except Exception as e:
            cleanup(runner_dir, err, message="Error initializing postgres")

        benchbase_run_results_dir = output_dir / str(run_id)

        try:
            init_benchbase(benchbase_dir, benchmark_name, benchmark_cfg_path)
        except Exception as err:
            cleanup(runner_dir, err, message="Error initializing benchbase")

        try:
            tscout_proc = init_tscout(tscout_dir, benchmark_name, experiment_name, run_id)
        except Exception as err:
            cleanup(runner_dir, err, message="Error initializing TScout")

        try:
            exec_benchbase(
                benchbase_dir, benchmark_name, benchbase_run_results_dir,
            )
        except Exception as err:
            cleanup(runner_dir, err, message="Error running benchbase")

        try:
            pg_proc.kill()
            tscout_proc.kill()
            pg_log_file.close()
        except Exception as err:
            cleanup(runner_dir, err, message="Error killing processes")

        print(f"Completed run {run_id}")
