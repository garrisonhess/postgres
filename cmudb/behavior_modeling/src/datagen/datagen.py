# allow plumbum FG statements
# pylint: disable=pointless-statement

import os
import shlex
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from subprocess import Popen
from typing import Any, Optional

import psutil
import yaml
from plumbum import FG, local
from plumbum.cmd import bash, make, sudo  # pylint: disable=import-error

from src import (
    BEHAVIOR_MODELING_DIR,
    BENCH_DBS,
    BENCH_TABLES,
    BENCHBASE_DIR,
    BENCHBASE_SNAPSHOT_DIR,
    BENCHBASE_SNAPSHOT_PATH,
    CLEANUP_SCRIPT_PATH,
    DATA_ROOT,
    PG_CONF_PATH,
    PG_DIR,
    SQLSMITH_DIR,
    TSCOUT_DIR,
    get_logger,
)


def build_pg() -> None:
    """Build Postgres (and extensions)"""

    try:
        os.chdir(PG_DIR)
        bash["./cmudb/build/configure.sh", "release"] & FG
        make["clean", "-s"]()
        make["-j", "world-bin", "-s"]()
        make["install-world-bin", "-j", "-s"]()
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error building postgres")


def check_orphans() -> None:
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

        if any((tscout_process_name in proc.info["name"] for tscout_process_name in tscout_process_names)):
            tscout_procs.append(proc)

    assert len(pg_procs) == 0, f"Found active postgres processes from previous runs: {pg_procs}"
    assert len(tscout_procs) == 0, f"Found active tscout processes from previous runs: {tscout_procs}"


def init_pg(auto_explain: bool, stat_statements: bool, store_plans: bool) -> None:
    try:
        os.chdir(PG_DIR)

        # initialize postgres for benchbase execution
        shutil.rmtree("data")

        Path("data").mkdir(parents=True, exist_ok=True)
        bash["./build/bin/pg_ctl", "initdb", "-D", "data"] & FG
        shutil.copy(str(PG_CONF_PATH), "./data/postgresql.conf")

        ctl_str = shlex.split("""./build/bin/pg_ctl -D data -o "-W 2" start""")
        bash[ctl_str] & FG

        # Initialize the DB and create an admin user for Benchbase to use
        bash["./build/bin/createdb", "test"] & FG
        role_cmd = shlex.split(
            '''./build/bin/psql -d test -c "CREATE ROLE admin WITH PASSWORD 'password' SUPERUSER CREATEDB CREATEROLE INHERIT LOGIN;"'''
        )
        bash[role_cmd] & FG
        bash["./build/bin/createdb", "-O", "admin", "benchbase"] & FG

        # Turn on QueryID computation
        queryid_cmd1 = shlex.split('''./build/bin/psql -d test -c "ALTER DATABASE test SET compute_query_id = 'ON';"''')
        bash[queryid_cmd1] & FG
        queryid_cmd2 = shlex.split(
            '''./build/bin/psql -d benchbase -c "ALTER DATABASE benchbase SET compute_query_id = 'ON';"'''
        )
        bash[queryid_cmd2] & FG

        if auto_explain:
            auto_explain_cmd = shlex.split(
                '''./build/bin/psql -d benchbase -c "ALTER SYSTEM SET auto_explain.log_min_duration = 0;"'''
            )
            bash[auto_explain_cmd] & FG
            bash["./build/bin/pg_ctl", "-D", "data", "reload"] & FG

        if stat_statements:
            enable_stat_stmt = shlex.split(
                '''./build/bin/psql -d 'benchbase' -c "CREATE EXTENSION pg_stat_statements;"'''
            )
            bash[enable_stat_stmt] & FG

        if store_plans:
            enable_store_plans = shlex.split(
                '''./build/bin/psql -d 'benchbase' -c "CREATE EXTENSION pg_store_plans;"'''
            )
            bash[enable_store_plans] & FG

        # Turn off pager
        disable_pager = shlex.split('''./build/bin/psql -d benchbase -P pager=off -c "SELECT 1;"''')
        bash[disable_pager] & FG

    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error initializing Postgres")


def pg_analyze(bench_db: str) -> None:
    try:
        os.chdir(PG_DIR)

        if bench_db not in BENCH_TABLES:
            raise ValueError(f"Benchmark {bench_db} doesn't have tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            get_logger().info("Analyzing table: %s", table)
            analyze_cmd = shlex.split(f"""./build/bin/psql -d benchbase -c 'ANALYZE VERBOSE {table};'""")
            bash[analyze_cmd] & FG
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error analyzing Postgres")


def pg_prewarm(bench_db: str) -> None:
    """Prewarm Postgres so the buffer pool and OS page cache has the workload data available"""

    try:
        os.chdir(PG_DIR)
        enable_prewarm = shlex.split('''./build/bin/psql -d benchbase -c "CREATE EXTENSION pg_prewarm"''')
        bash[enable_prewarm] & FG

        if bench_db not in BENCH_TABLES:
            raise ValueError(f"Benchmark {bench_db} doesn't have prewarm tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            get_logger().info("Prewarming table: %s", table)
            prewarm_cmd = shlex.split(f"""./build/bin/psql -d benchbase -c "select * from pg_prewarm('{table}')";""")
            bash[prewarm_cmd] & FG
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error prewarming Postgres")


def init_tscout(results_dir: Path) -> Popen[bytes]:
    try:
        os.chdir(TSCOUT_DIR)

        tscout_results_dir = results_dir / "tscout"
        tscout_results_dir.mkdir(exist_ok=True)
        # Make sure we're authed with sudo before running tscout in background
        Popen(args=["sudo pwd"], shell=True).wait()
        tscout_proc = Popen(
            args=[f"sudo python3 tscout.py `pgrep -ox postgres` --outdir {tscout_results_dir} &"],
            shell=True,
        )
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error initializing TScout")

    time.sleep(1)  # allows tscout to attach before Benchbase execution begins
    return tscout_proc


def build_benchbase(benchbase_dir: Path) -> None:
    get_logger().info("Building Benchbase")

    try:
        os.chdir(benchbase_dir)
        bash["./mvnw", "clean", "package"] & FG
        if not os.path.exists(BENCHBASE_SNAPSHOT_DIR):
            local["unzip"][BENCHBASE_SNAPSHOT_PATH] & FG
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error building benchbase")


def init_benchbase(bench_db: str, benchbase_results_dir: Path) -> None:
    """Initialize Benchbase and load benchmark data"""
    logger = get_logger()

    try:
        os.chdir(BENCHBASE_DIR)
        if not os.path.exists(BENCHBASE_SNAPSHOT_DIR):
            build_benchbase(BENCHBASE_DIR)
        os.chdir(BENCHBASE_SNAPSHOT_DIR)

        # move runner config to benchbase and also save it in the output directory
        input_cfg_path = BEHAVIOR_MODELING_DIR / f"config/datagen/benchbase/{bench_db}_config.xml"
        benchbase_cfg_path = BENCHBASE_SNAPSHOT_DIR / f"config/postgres/{bench_db}_config.xml"
        shutil.copy(input_cfg_path, benchbase_cfg_path)
        shutil.copy(input_cfg_path, benchbase_results_dir)

        print("initializing benchbase")
        logger.info("Initializing Benchbase for DB: %s", bench_db)
        benchbase_cmd = [
            "-jar",
            "benchbase.jar",
            "-b",
            bench_db,
            "-c",
            f"config/postgres/{bench_db}_config.xml",
            "--create=true",
            "--load=true",
            "--execute=false",
        ]
        local["java"][benchbase_cmd] & FG
        logger.info("Initialized Benchbase for Benchmark: %s", bench_db)
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error initializing Benchbase")


def exec_benchbase(bench_db: str, results_dir: Path, benchbase_results_dir: Path, config: dict[str, Any]) -> None:
    psql_path = PG_DIR / "./build/bin/psql"

    try:
        os.chdir(BENCHBASE_DIR)
        if not os.path.exists(BENCHBASE_SNAPSHOT_DIR):
            build_benchbase(BENCHBASE_DIR)
        os.chdir(BENCHBASE_SNAPSHOT_DIR)

        if config["pg_stat_statements"]:
            bash[psql_path, "-d", "'benchbase'", "-c", "SELECT pg_stat_statements_reset();"] & FG
        if config["pg_store_plans"]:
            bash[psql_path, "-d", "'benchbase'", "-c", "SELECT pg_store_plans_reset();"] & FG

        # run benchbase
        benchbase_cmd = [
            "-jar",
            "benchbase.jar",
            "-b",
            bench_db,
            "-c",
            f"config/postgres/{bench_db}_config.xml",
            "--create=false",
            "--load=false",
            "--execute=true",
        ]
        local["java"](benchbase_cmd)

        if config["pg_stat_statements"]:
            with (results_dir / "stat_file.csv").open("w") as f:
                stats_cmd = [psql_path, "-d", """'benchbase'""" "--csv", "-c", "SELECT * FROM pg_stat_statements;"]
                stats_result = bash[stats_cmd]()
                f.write(stats_result)

        if config["pg_store_plans"]:
            with (results_dir / "plan_file.csv").open("w") as f:
                plans_query = "SELECT queryid, planid, plan FROM pg_store_plans ORDER BY queryid, planid"
                plans_result = bash[psql_path](["-d", "'benchbase'", "--csv", "-c", plans_query])
                f.write(plans_result)

        # Move benchbase results to experiment results directory
        shutil.move(BENCHBASE_SNAPSHOT_DIR / "results", benchbase_results_dir)
        time.sleep(5)  # Allow TScout Collector to finish getting results
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error running Benchbase")


def cleanup(err: Optional[RuntimeError], terminate: bool, message: str = "") -> None:
    """Clean up the TScout and Postgres processes after either a successful or failed run"""

    logger = get_logger()

    if len(message) > 0:
        logger.error(message)

    if err is not None:
        logger.error("Error: %s", err)

    username = psutil.Process().username()
    sudo[["python3", CLEANUP_SCRIPT_PATH, "--username", username]] & FG
    time.sleep(2)  # Allow TScout poison pills to propagate

    # Exit the program if the caller requested it (only happens on error)
    if terminate:
        sys.exit(1)


def exec_sqlsmith(bench_db: str) -> None:

    try:
        os.chdir(PG_DIR)
        # Add SQLSmith user to benchbase DB with non-superuser privileges

        role_cmd = shlex.split(
            '''./build/bin/psql -d benchbase -c "CREATE ROLE sqlsmith WITH PASSWORD 'password' INHERIT LOGIN;"'''
        )
        bash[role_cmd] & FG

        if bench_db not in BENCH_TABLES:
            raise ValueError(f"Benchmark {bench_db} doesn't have tables setup yet.")

        for table in BENCH_TABLES[bench_db]:
            get_logger().info("Granting SQLSmith permissions on table: %s", table)
            perm_cmd = shlex.split(
                f'''./build/bin/psql -d benchbase -c "GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO sqlsmith;"'''
            )
            bash[perm_cmd] & FG

        os.chdir(SQLSMITH_DIR)
        sqlsmith_cmd = shlex.split(
            """./sqlsmith --target="host=localhost port=5432 dbname=benchbase connect_timeout=10" --seed=42 --max-queries=10000 --exclude-catalog"""
        )
        bash[sqlsmith_cmd] & FG
    except RuntimeError as err:
        cleanup(err, terminate=True, message="Error running SQLSmith")


def run(bench_db: str, results_dir: Path, benchbase_results_dir: Path, config: dict[str, Any]) -> None:
    """Run an experiment"""
    assert results_dir.exists(), f"Results directory does not exist: {results_dir}"

    check_orphans()

    init_pg(config["auto_explain"], config["pg_stat_statements"], config["pg_store_plans"])
    init_benchbase(bench_db, benchbase_results_dir)

    # reload config to make a new logfile
    os.chdir(PG_DIR)
    smart_shutdown = shlex.split("./build/bin/pg_ctl stop -D data -m smart")
    bash[smart_shutdown] & FG

    # remove pre-existing logs
    for log_path in [fp for fp in (PG_DIR / "data/log").glob("*") if fp.suffix in ["csv", "log"]]:
        log_path.unlink()

    ctl_cmd: list[str] = shlex.split("""./build/bin/pg_ctl -D data -o "-W 2" start""")
    bash[ctl_cmd] & FG

    if config["pg_prewarm"]:
        pg_analyze(bench_db)
        pg_prewarm(bench_db)

    tscout_proc = init_tscout(results_dir)
    exec_benchbase(bench_db, results_dir, benchbase_results_dir, config)

    log_fps = list((PG_DIR / "data/log").glob("*.log"))
    assert len(log_fps) == 1, f"Expected 1 Postgres log file, found {len(log_fps)}, {log_fps}"
    shutil.move(str(log_fps[0]), str(results_dir))
    log_fps = list(results_dir.glob("*.log"))
    assert len(log_fps) == 1, f"Expected 1 Result log file, found {len(log_fps)}, {log_fps}"
    log_fps[0].rename(results_dir / "pg_log.log")

    cleanup(err=None, terminate=False, message="Finished run")
    tscout_proc.wait()


def main(config_name: str) -> None:
    config_path = BEHAVIOR_MODELING_DIR / f"config/datagen/{config_name}.yaml"
    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger = get_logger()

    # validate the benchmark databases from the config
    bench_dbs = config["bench_dbs"]
    for bench_db in bench_dbs:
        if bench_db not in BENCH_DBS:
            raise ValueError(f"Invalid benchmark database: {bench_db}")

    if config["build_pg"]:
        build_pg()

    if config["build_bbase"]:
        build_benchbase(BENCHBASE_DIR)

    # Setup experiment directory
    experiment_name = f"experiment-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    modes = ["train", "eval"] if not config["debug"] else ["debug"]

    for mode in modes:
        mode_dir = DATA_ROOT / mode / experiment_name

        # copy datagen configuration to the output directory
        Path(mode_dir).mkdir(parents=True)
        shutil.copy(config_path, mode_dir)

        for bench_db in bench_dbs:
            results_dir = mode_dir / bench_db
            Path(results_dir).mkdir()
            benchbase_results_dir = results_dir / "benchbase"
            Path(benchbase_results_dir).mkdir(exist_ok=True)
            logger.info(
                "Running experiment: %s with bench_db: %s and results_dir: %s", experiment_name, bench_db, results_dir
            )
            run(bench_db, results_dir, benchbase_results_dir, config)
