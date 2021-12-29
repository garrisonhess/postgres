import pandas as pd
from config import TRAIN_DATA_ROOT, DATA_ROOT
from collections import defaultdict
import json


class PlanTree:
    def __init__(self):
        self.x = 1


# get latest experiment
experiment_list = sorted([exp_path.name for exp_path in TRAIN_DATA_ROOT.glob("*")])
assert len(experiment_list) > 0, "No experiments found"
experiment = experiment_list[-1]
# print(f"Differencing latest experiment: {experiment}")

results_dir = DATA_ROOT / "train" / experiment / "tpcc"
plan_file_path = results_dir / "plan_file.csv"

plan_df = pd.read_csv(plan_file_path)


print(plan_df.head(1))


print(plan_df.columns)


# ['userid', 'dbid', 'queryid', 'planid', 'plan', 'calls', 'total_time',
#  'min_time', 'max_time', 'mean_time', 'stddev_time', 'rows',
#  'shared_blks_hit', 'shared_blks_read', 'shared_blks_dirtied',
#  'shared_blks_written', 'local_blks_hit', 'local_blks_read',
#  'local_blks_dirtied', 'local_blks_written', 'temp_blks_read',
#  'temp_blks_written', 'blk_read_time', 'blk_write_time', 'first_call',
#  'last_call']


query_id_to_all_plans = defaultdict(list)
plan_df = plan_df[["queryid", "planid", "plan"]]


for i in range(len(plan_df.index)):
    qid = plan_df.iloc[i]["queryid"]
    plan_id = plan_df.iloc[i]["planid"]
    plan = plan_df.iloc[i]["plan"]
    query_id_to_all_plans[qid].append((plan_id, plan))


query_id_to_plans = defaultdict(tuple)

for query_id, plans in query_id_to_all_plans.items():

    if len(plans) > 1:
        print(f"query_id: {query_id}, num_plans: {len(plans)}")
        # print(f"PLAN 1: {plans[0]}")
        # print(f"PLAN 2: {plans[1]}")

        (plan_id1, plan1) = plans[0]
        (plan_id2, plan2) = plans[1]
        plan1 = json.loads(plan1)
        plan2 = json.loads(plan2)

        print(plan1)

        print(type(plans[0][1]))


        break
    else:
        query_id_to_plans[query_id] = plans[0]
