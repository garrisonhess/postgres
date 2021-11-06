from subprocess import PIPE, Popen
import time
import psutil
import os
import argparse
from pathlib import Path


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

    # pg_logfile_path = results_dir / "pg_runner_logs.txt"
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


def init_tscout(tscout_dir):
    print(f"Starting TScout")
    os.chdir(tscout_dir)
    Popen(args=["rm -f *.csv"], shell=True).wait()
    Popen(args=["sudo pwd"], shell=True).wait()
    Popen(args=["sudo python3 tscout.py `pgrep -ox postgres` &"], shell=True).wait()


def init_benchbase(build_benchbase, benchbase_dir):
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
    print("Starting Benchbase")
    benchbase_cmd = "java -jar benchbase.jar -b tpcc -c config/postgres/sample_tpcc_config.xml --create=true --load=true"
    Popen(
        args=[benchbase_cmd],
        shell=True,
    ).wait()


def shutdown(pg_proc):
    # for proc in psutil.process_iter(["pid", "name", "username"]):
    #     if proc.info["name"].lower().startswith("postgres"):
    #         try:
    #             proc.kill()
    #         except (psutil.NoSuchProcess, psutil.ZombieProcess):
    #             pass

    # logfile_name = "/home/gh/postgres/cmu-db/tscout/"
    # if os.path.exists(logfile_name):
    #     os.remove(logfile_name)
    # logfile = open(logfile_name, "w")
    # logfile.close()

    # Shutdown postgres, tscout, and benchbase
    print("Shutting down PG process and closing logfile")
    pg_proc.kill()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Postgres/Benchbase/TScout")
    parser.add_argument(
        "--build-benchbase", dest="build_benchbase", action="store_true", default=False
    )
    parser.add_argument(
        "--build-pg", dest="build_pg", action="store_true", default=False
    )

    args = parser.parse_args()
    results_dir = Path.home() / "cmudb" / "modeling" / "results"

    pg_dir = Path.home() / "postgres"
    pg_proc = init_pg(args.build_pg, pg_dir, results_dir)

    tscout_dir = tscout_dir = pg_dir / "cmudb" / "tscout"
    init_tscout(tscout_dir)

    benchbase_dir = benchbase_dir = Path.home() / "benchbase"
    init_benchbase(args.build_benchbase, benchbase_dir)

    shutdown(pg_proc)
