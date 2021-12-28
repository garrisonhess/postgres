#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen

import psutil
import yaml

from config import BENCH_DBS, BENCH_TABLES, DATA_ROOT

logger = logging.getLogger("datagen")
logger.setLevel("INFO")
pg_dir = Path.home() / "postgres"
cmudb_dir = pg_dir / "cmudb"
tscout_dir = cmudb_dir / "tscout"
benchbase_dir = Path.home() / "benchbase"
benchbase_snapshot_dir = benchbase_dir / "benchbase-2021-SNAPSHOT"
benchbase_snapshot_path = benchbase_dir / "target" / "benchbase-2021-SNAPSHOT.zip"
behavior_modeling_dir = cmudb_dir / "behavior_modeling"
pg_conf_path = behavior_modeling_dir / "config/datagen/postgres/postgresql.conf"
cleanup_script_path = behavior_modeling_dir / "cleanup.py"
sqlsmith_dir = Path.home() / "sqlsmith"


def build_pg():
    """Build Postgres (and extensions)"""

    try:
        os.chdir(pg_dir)
        Popen(args=["./cmudb/build/configure.sh release"], shell=True).wait()
        Popen(["make clean -s"], shell=True).wait()
        Popen(args=["make -j world-bin -s"], shell=True).wait()
        Popen(args=["make install-world-bin -j -s"], shell=True).wait()
    except Exception as err:
        cleanup(err, terminate=True, message="Error building postgres")


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


def init_pg():
    try:
        os.chdir(pg_dir)

        # initialize postgres for benchbase execution
        shutil.rmtree("data")

        Path("data").mkdir(parents=True, exist_ok=True)
        Popen(args=["./build/bin/pg_ctl initdb -D data"], shell=True).wait()
        shutil.copy(str(pg_conf_path), "./data/postgresql.conf")

        Popen(
            args=["""./build/bin/pg_ctl -D data -o "-W 2" start"""],
            shell=True,
        ).wait()

        # Initialize the DB and create an admin user for Benchbase to use
        Popen(args=["./build/bin/createdb test"], shell=True).wait()
        Popen(
            args=[
                '''./build/bin/psql -d test -c "CREATE ROLE admin WITH PASSWORD 'password' SUPERUSER CREATEDB CREATEROLE INHERIT LOGIN;"'''
            ],
            shell=True,
        ).wait()
        Popen(args=["./build/bin/createdb -O admin benchbase"], shell=True).wait()

        # Turn on QueryID computation
        Popen(
            args=['''./build/bin/psql -d test -c "ALTER DATABASE test SET compute_query_id = 'ON';"'''],
            shell=True,
        ).wait()
        Popen(
            args=['''./build/bin/psql -d benchbase -c "ALTER DATABASE benchbase SET compute_query_id = 'ON';"'''],
            shell=True,
        ).wait()

        if config["pg_stat_statements"]:
            Popen(
                args=['''./build/bin/psql -d 'benchbase' -c "CREATE EXTENSION pg_stat_statements;"'''], shell=True
            ).wait()
        if config["pg_store_plans"]:
            Popen(args=['''./build/bin/psql -d 'benchbase' -c "CREATE EXTENSION pg_store_plans;"'''], shell=True).wait()

        # Turn off pager
        Popen(
            args=['''./build/bin/psql -d benchbase -P pager=off -c "SELECT 1;"'''],
            shell=True,
        ).wait()

    except Exception as err:
        cleanup(err, terminate=True, message="Error initializing Postgres")


def pg_analyze(bench_db):
    try:
        os.chdir(pg_dir)

        if bench_db not in BENCH_TABLES.keys():
            raise ValueError(f"Benchmark {bench_db} doesn't have tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            logger.info(f"Analyzing table: {table}")
            Popen(
                args=[f"""./build/bin/psql -d benchbase -c 'ANALYZE VERBOSE {table};'"""],
                shell=True,
            ).wait()
    except Exception as err:
        cleanup(err, terminate=True, message="Error analyzing Postgres")


def pg_prewarm(bench_db):
    """Prewarm Postgres so the buffer pool and OS page cache has the workload data available"""

    try:
        os.chdir(pg_dir)
        Popen(
            args=['''./build/bin/psql -d benchbase -c "CREATE EXTENSION pg_prewarm"'''],
            shell=True,
        ).wait()

        if bench_db not in BENCH_TABLES.keys():
            raise ValueError(f"Benchmark {bench_db} doesn't have prewarm tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            logger.info(f"Prewarming table: {table}")
            Popen(
                args=[f"""./build/bin/psql -d benchbase -c "select * from pg_prewarm('{table}')";"""],
                shell=True,
            ).wait()
    except Exception as err:
        cleanup(err, terminate=True, message="Error prewarming Postgres")


def init_tscout(results_dir):
    try:
        os.chdir(tscout_dir)
        # Make sure we're authed with sudo before running tscout in background
        Popen(args=["sudo pwd"], shell=True).wait()
        tscout_proc = Popen(
            args=[f"sudo python3 tscout.py `pgrep -ox postgres` --outdir {results_dir} &"],
            shell=True,
        )
    except Exception as err:
        cleanup(err, terminate=True, message="Error initializing TScout")

    time.sleep(1)  # allows tscout to attach before Benchbase execution begins
    return tscout_proc


def build_benchbase(benchbase_dir):
    logger.info("Building Benchbase")

    try:
        os.chdir(benchbase_dir)
        Popen(args=["./mvnw clean package"], shell=True).wait()

        # TODO: resolve this in a cleaner way
        if not os.path.exists(benchbase_snapshot_dir):
            Popen(args=[f"unzip {benchbase_snapshot_path}"], shell=True).wait()
    except Exception as err:
        cleanup(err, terminate=True, message="Error building benchbase")


def init_benchbase(bench_db):
    """Initialize Benchbase and load benchmark data"""

    try:
        os.chdir(benchbase_dir)
        if not os.path.exists(benchbase_snapshot_dir):
            build_benchbase(benchbase_dir)
        os.chdir(benchbase_snapshot_dir)

        # move runner config to benchbase and also save it in the output directory
        input_cfg_path = behavior_modeling_dir / f"config/datagen/benchbase/{bench_db}_config.xml"
        benchbase_cfg_path = benchbase_snapshot_dir / f"config/postgres/{bench_db}_config.xml"
        shutil.copy(input_cfg_path, benchbase_cfg_path)
        shutil.copy(input_cfg_path, benchbase_results_dir)

        logger.info(f"Initializing Benchbase for DB: {bench_db}")
        benchbase_cmd = f"java -jar benchbase.jar -b {bench_db} -c config/postgres/{bench_db}_config.xml --create=true --load=true --execute=false"
        bbase_proc = Popen(args=[benchbase_cmd], shell=True)
        bbase_proc.wait()
        if bbase_proc.returncode != 0:
            raise RuntimeError(f"Benchbase failed with return code: {bbase_proc.returncode}")
        logger.info(f"Initialized Benchbase for Benchmark: {bench_db}")
    except Exception as err:
        cleanup(err, terminate=True, message="Error initializing Benchbase")


def exec_benchbase(bench_db):
    psql_path = pg_dir / "./build/bin/psql"

    try:
        os.chdir(benchbase_dir)
        if not os.path.exists(benchbase_snapshot_dir):
            build_benchbase(benchbase_dir)
        os.chdir(benchbase_snapshot_dir)

        if config["pg_stat_statements"]:
            Popen(args=[f'''{psql_path} -d 'benchbase' -c "SELECT pg_stat_statements_reset();"'''], shell=True).wait()

        if config["pg_store_plans"]:
            Popen(args=[f'''{psql_path} -d 'benchbase' -c "SELECT pg_store_plans_reset();"'''], shell=True).wait()

        # run benchbase
        benchbase_cmd = f"java -jar benchbase.jar -b {bench_db} -c config/postgres/{bench_db}_config.xml --create=false --load=false --execute=true"
        bbase_proc = Popen(args=[benchbase_cmd], shell=True)
        bbase_proc.wait()
        if bbase_proc.returncode != 0:
            raise RuntimeError(f"Benchbase failed with return code: {bbase_proc.returncode}")

        if config["pg_stat_statements"]:
            with open(results_dir / "stat_file.csv", "w") as stat_file:
                Popen(
                    args=[f'''{psql_path} -d 'benchbase' --csv -c "SELECT * FROM pg_stat_statements;"'''],
                    shell=True,
                    stdout=stat_file,
                    stderr=stat_file,
                ).wait()
        if config["pg_store_plans"]:
            with open(results_dir / "plan_file.csv", "w") as stat_file:
                Popen(
                    args=[f'''{psql_path} -d 'benchbase' --csv -c "SELECT * FROM pg_store_plans;"'''],
                    shell=True,
                    stdout=stat_file,
                    stderr=stat_file,
                ).wait()

        # Move benchbase results to experiment results directory
        shutil.move(str(benchbase_snapshot_dir / "results"), str(benchbase_results_dir))
        time.sleep(5)  # Allow TScout Collector to finish getting results

    except Exception as err:
        cleanup(err, terminate=True, message="Error running Benchbase")


def cleanup(err, terminate, message=""):
    """Clean up the TScout and Postgres processes after either a successful or failed run"""

    if len(message) > 0:
        logger.error(message)

    if err is not None:
        logger.error(f"Error: {err}")

    username = psutil.Process().username()
    Popen(args=[f"sudo python3 {cleanup_script_path} --username {username}"], shell=True).wait()
    time.sleep(2)  # Allow TScout poison pills to propagate

    # Exit the program if the caller requested it (only happens on error)
    if terminate:
        exit(1)


def exec_sqlsmith(bench_db):

    try:
        os.chdir(pg_dir)
        # Add SQLSmith user to benchbase DB with non-superuser privileges
        Popen(
            args=[
                '''./build/bin/psql -d benchbase -c "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;"'''
            ],
            shell=True,
        ).wait()

        if bench_db not in BENCH_TABLES.keys():
            raise ValueError(f"Benchmark {bench_db} doesn't have tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            logger.info(f"Granting SQLSmith permissions on table: {table}")
            Popen(
                args=[
                    f'''./build/bin/psql -d benchbase -c "GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO sqlsmith;"'''
                ],
                shell=True,
            ).wait()

        os.chdir(sqlsmith_dir)
        sqlsmith_cmd = """./sqlsmith --target="host=localhost port=5432 dbname=benchbase connect_timeout=10" --seed=42 --max-queries=10000 --exclude-catalog"""
        Popen(args=[sqlsmith_cmd], shell=True).wait()
    except Exception as err:
        cleanup(err, terminate=True, message="Error running SQLSmith")


def run(bench_db, results_dir):
    """Run an experiment"""
    assert results_dir.exists(), f"Results directory does not exist: {results_dir}"
    logger.info(f"Running experiment: {experiment_name} with bench_db: {bench_db} and results_dir: {results_dir}")

    check_orphans()

    init_pg()
    init_benchbase(bench_db)

    # reload config to make a new logfile
    os.chdir(pg_dir)
    Popen(args=["./build/bin/pg_ctl -D data stop"], shell=True).wait()

    # remove pre-existing logs
    for log_path in [fp for fp in (pg_dir / "data/log").glob("*") if fp.suffix in ["csv", "log"]]:
        log_path.unlink()

    Popen(args=["""./build/bin/pg_ctl -D data -o "-W 2" start"""], shell=True).wait()

    if config["pg_prewarm"]:
        pg_analyze(bench_db)
        pg_prewarm(bench_db)

    tscout_proc = init_tscout(results_dir)
    exec_benchbase(bench_db)

    log_fps = list((pg_dir / "data/log").glob("*.log"))
    assert len(log_fps) == 1, f"Expected 1 log file, found {len(log_fps)}"
    shutil.move(str(log_fps[0]), str(results_dir))
    log_fps = list(results_dir.glob("*.log"))
    assert len(log_fps) == 1, f"Expected 1 log file, found {len(log_fps)}"
    log_fps[0].rename(results_dir / "pg_log.log")

    cleanup(err=None, terminate=False, message="Finished run")
    tscout_proc.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment with Postgres, Benchbase, and TScout")
    parser.add_argument("--config", type=str, default="default")
    args = parser.parse_args()
    config_name = args.config

    # Load datagen config
    config_path = behavior_modeling_dir / f"config/datagen/{config_name}.yaml"
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    logger.setLevel(config["log_level"])

    # validate the benchmark databases from the config
    bench_dbs = config["bench_dbs"]
    for bench_db in bench_dbs:
        if bench_db not in BENCH_DBS:
            raise ValueError(f"Invalid benchmark database: {bench_db}")

    if config["build_pg"]:
        build_pg()

    if config["build_bbase"]:
        build_benchbase(benchbase_dir)

    # Setup experiment directory
    experiment_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    for mode in ["train", "eval"]:
        mode_dir = DATA_ROOT / mode / experiment_name

        for bench_db in bench_dbs:
            results_dir = mode_dir / bench_db
            Path(results_dir).mkdir(parents=True, exist_ok=True)
            benchbase_results_dir = results_dir / "benchbase"
            Path(benchbase_results_dir).mkdir(exist_ok=True)
            run(bench_db, results_dir)
