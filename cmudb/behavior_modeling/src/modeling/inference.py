#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path


def load_models():
    models_root = Path("./models")

    if not models_root.exists():
        raise ValueError(f"models_root: {models_root} doesn't exist")

    models = list(models_root.iterdir())
    if len(models) == 0:
        raise ValueError(f"No models in models_root: {models_root}")

    model_base_dir = models[0]
    model_dir = model_base_dir.iterdir()[0]
    ou_to_model = dict()

    for ou_model_dir in model_dir.iterdir():
        ou_name = ou_model_dir.stem
        model_pickle = list(ou_model_dir.glob(f"*_{ou_name}.pkl"))
        if len(model_pickle) > 1:
            raise ValueError(f"Found more than 1 model pickle in ou_model_dir: {ou_model_dir}")

        model_pickle = model_pickle[0]

        with open(model_pickle, "r") as f:
            print(f"loading model pickle: {model_pickle}")
            ou_to_model[ou_name] = pickle.load(f)

    return ou_to_model


# Inference Steps:
# 1. Load models
# 2. Load data
# 3. Run inference and write to disk
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OU Model Inference Runner")
    parser.add_argument("--output_dir", type=str, default="./inference_results/")
    parser.add_argument("--input_dir", type=str, default="./training_data/")
    args = parser.parse_args()

    indir = Path(args.input_dir)
    outdir = Path(args.output_dir)

    ou_to_model = load_models()
    # data = load_data()
    # infer(ou_to_model, data)
