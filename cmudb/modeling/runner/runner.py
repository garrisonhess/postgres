#!/usr/bin/env python3

from subprocess import PIPE, Popen
import time
import os
import argparse
from pathlib import Path
import shutil
from datetime import datetime
import psutil

# TODO: change psql commands to use psycopg2
def init_pg(build_pg, pg_dir, output_dir):
    os.chdir(pg_dir)
    pg_log_file = open(output_dir / "pg_log.txt", "w")

    if build_pg:
        print("Building Postgres")
        Popen(args=["./cmudb/build/configure.sh debug"], shell=True).wait()
        Popen(["make clean -s"], shell=True).wait()
        Popen(args=["make -j -s"], shell=True).wait()
        Popen(args=["make install -j -s"], shell=True).wait()

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
        args=[
            '''./build/bin/psql -d test -c "ALTER DATABASE test SET compute_query_id = 'ON';"'''
        ],
        shell=True,
    ).wait()
    Popen(
        args=[
            '''./build/bin/psql -d benchbase -c "ALTER DATABASE benchbase SET compute_query_id = 'ON';"'''
        ],
        shell=True,
    ).wait()

    return pg_proc, pg_log_file


# TODO: replace pgrep call with psutil
def init_tscout(tscout_dir, benchmark_name, experiment_name, run_id):
    print("Starting TScout")
    os.chdir(tscout_dir)
    tscout_proc = Popen(
        args=[
            f"sudo python3 tscout.py `pgrep -ox postgres` {benchmark_name} {experiment_name} {run_id} &"
        ],
        shell=True,
    )
    time.sleep(5)
    return tscout_proc


def run_benchbase(
    build_benchbase, benchbase_dir, benchmark_name, input_cfg_path, output_dir
):
    os.chdir(benchbase_dir)

    benchbase_snapshot_path = benchbase_dir / "target" / "benchbase-2021-SNAPSHOT.zip"
    benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"

    # build benchbase and setup tpc-c
    if build_benchbase:
        print(f"Building Benchbase from dir: {os.getcwd()}")
        Popen(args=["./mvnw clean package"], shell=True).wait()

    if not os.path.exists(benchbase_snapshot_dir):
        Popen(args=[f"unzip {benchbase_snapshot_path}"], shell=True).wait()

    os.chdir(benchbase_snapshot_dir)

    # move runner config to benchbase and also save it in the output directory
    benchbase_cfg_path = (
        benchbase_snapshot_dir / f"config/postgres/{benchmark_name}_config.xml"
    )
    shutil.copy(input_cfg_path, benchbase_cfg_path)
    shutil.copy(input_cfg_path, output_dir)

    print("Starting Benchbase")
    benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=true --load=true --execute=true"
    Popen(args=[benchbase_cmd], shell=True).wait()

    # Copy benchbase results to experimental results directory
    benchbase_results_dir = benchbase_snapshot_dir / "results"
    print(f"Copying Benchbase results from {benchbase_results_dir} to {output_dir}")
    result_files = benchbase_results_dir.glob("**/*")
    for file in result_files:
        print(f"Copying {file} to {output_dir}")
        shutil.copy(file, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Postgres/Benchbase/TScout")
    parser.add_argument(
        "--build-benchbase", dest="build_benchbase", action="store_true", default=False
    )
    parser.add_argument(
        "--build-pg", dest="build_pg", action="store_true", default=False
    )
    parser.add_argument(
        "--benchmark-name", dest="benchmark_name", action="store", default="tpcc"
    )
    parser.add_argument(
        "--experiment-name", dest="experiment_name", action="store", default=""
    )
    parser.add_argument("--nruns", dest="nruns", action="store", default=5)

    args = parser.parse_args()
    build_benchbase = args.build_benchbase
    benchmark_name = args.benchmark_name
    experiment_name = args.experiment_name
    build_pg = args.build_pg
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
    benchmark_cfg_path = (
        runner_dir / "benchbase_configs" / f"{benchmark_name}_config.xml"
    )
    benchbase_dir = Path.home() / "benchbase"
    output_dir = tscout_dir / "results" / benchmark_name / experiment_name
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"Running experiment: {experiment_name} with {nruns} runs and output_dir: {output_dir}"
    )
    for run_id in range(nruns):
        print(f"Starting run {run_id}")
        pg_proc, pg_log_file = init_pg(build_pg, pg_dir, output_dir)
        tscout_proc = init_tscout(tscout_dir, benchmark_name, experiment_name, run_id)
        run_benchbase(
            build_benchbase,
            benchbase_dir,
            benchmark_name,
            benchmark_cfg_path,
            output_dir,
        )

        pg_proc.kill()
        tscout_proc.kill()
        pg_log_file.close()
        cleanup_script_path = runner_dir / "cleanup_run.py"
        username = psutil.Process().username()
        Popen(
            args=[f"sudo python3 {cleanup_script_path} {username}"], shell=True
        ).wait()
