import argparse
from pathlib import Path

import numpy as np
from sklearn import model_selection

from . import model

np.set_printoptions(precision=4)
np.set_printoptions(edgeitems=10)
np.set_printoptions(suppress=True)


class OUModelTrainer:
    """
    Trainer for the ou models
    """

    def __init__(
        self, input_path,
    ):
        self.input_path = input_path
        self.stats_map = {}

    def train(self):
        pass


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="OU Model Trainer")
    aparser.add_argument("--log", default="info", help="The logging level")
    args = aparser.parse_args()

    rf_model = model.Model(method="rf", normalize=False, log_transform=False,)

    rf_model.train(x, y)
    rf_model.predict(x)
