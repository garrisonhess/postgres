import argparse

from src.datagen import datagen

# from src.modeling import train
# from src.plans import diff

parser = argparse.ArgumentParser(description="Run an experiment with Postgres, Benchbase, and TScout")
parser.add_argument("--config", type=str, default="default")
args = parser.parse_args()
config_name = args.config

datagen.main(config_name)
# diff.main("default")
# train.main("default")
