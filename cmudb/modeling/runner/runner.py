#!/usr/bin/env python3

from subprocess import PIPE, Popen
import time
import psutil
import os
import argparse
from pathlib import Path
import shutil


def init_pg(build_pg, pg_dir, results_dir):
    os.chdir(pg_dir)

    if build_pg:
        print("Building Postgres")
        Popen(args=["./cmudb/build/configure.sh debug"], shell=True).wait()
        Popen(["make clean -s"], shell=True).wait()
        Popen(args=["make -j -s"], shell=True).wait()
        Popen(args=["make install -j -s"], shell=True).wait()

    # initialize postgres for benchbase execution
    Popen(args=["rm -r data"], shell=True).wait()
    Popen(args=["mkdir -p data"], shell=True).wait()
    Popen(args=["./build/bin/initdb -D data"], shell=True).wait()

    print(f"Starting Postgres")
    pg_proc = Popen(
        args=["./build/bin/postgres -D data -W 2 &"],
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
    )
    time.sleep(5)
    print(f"Started Postgres")

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

    return pg_proc


def init_tscout(tscout_dir, benchmark_name):
    print(f"Starting TScout")
    os.chdir(tscout_dir)
    Popen(args=["rm -f *.csv"], shell=True).wait()
    Popen(args=["sudo pwd"], shell=True).wait()
    Popen(
        args=[f"sudo python3 tscout.py `pgrep -ox postgres` {benchmark_name} &"],
        shell=True,
    )
    time.sleep(5)


def run_benchbase(build_benchbase, benchbase_dir, benchmark_name, input_cfg_path):
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
    print(os.getcwd())

    # move runner config to benchbase
    benchbase_cfg_path = (
        benchbase_snapshot_dir / f"config/postgres/{benchmark_name}_config.xml"
    )
    shutil.copyfile(input_cfg_path, benchbase_cfg_path)

    print("Starting Benchbase")
    benchbase_cmd = f"java -jar benchbase.jar -b {benchmark_name} -c config/postgres/{benchmark_name}_config.xml --create=true --load=true"
    Popen(
        args=[benchbase_cmd],
        shell=True,
    ).wait()


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

    args = parser.parse_args()
    build_benchbase = args.build_benchbase
    benchmark_name = args.benchmark_name
    build_pg = args.build_pg
    valid_benchmarks = ["tpcc", "tpch"]

    if benchmark_name not in valid_benchmarks:
        raise ValueError(f"Invalid benchmark name: {benchmark_name}")

    pg_dir = Path.home() / "postgres"
    cmudb_dir = pg_dir / "cmudb"
    tscout_dir = cmudb_dir / "tscout"
    modeling_dir = cmudb_dir / "modeling"
    results_dir = modeling_dir / "results" / benchmark_name
    runner_dir = modeling_dir / "runner"
    benchmark_cfg_path = (
        runner_dir / "benchbase_configs" / f"{benchmark_name}_config.xml"
    )
    benchbase_dir = benchbase_dir = Path.home() / "benchbase"

    pg_proc = init_pg(build_pg, pg_dir, results_dir)
    init_tscout(tscout_dir, benchmark_name)
    run_benchbase(build_benchbase, benchbase_dir, benchmark_name, benchmark_cfg_path)

    cleanup_script_path = runner_dir / "cleanup_run.py"
    Popen(args=[f"sudo python3 {cleanup_script_path}"], shell=True).wait()
    pg_proc.kill()
