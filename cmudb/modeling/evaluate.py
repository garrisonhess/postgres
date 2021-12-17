#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn import tree
import itertools
import model
import pydotplus


# if method == "dt":
#     for idx, target_name in enumerate(target_cols):
#         dot = tree.export_graphviz(ou_model._base_model.estimators_[idx], feature_names=feat_cols, filled=True)
#         dt_file = f"{ou_eval_dir}/{ou_name}_treeplot_{target_name}.png"
#         pydotplus.graphviz.graph_from_dot_data(dot).write_png(dt_file)


# y_train_pred = ou_model.predict(X_train)
# y_test_pred = ou_model.predict(X_test)

# # pair and reorder the target columns for readable outputs
# paired_cols = zip([f"pred_{col}" for col in target_cols], target_cols)
# reordered_cols = feat_cols + list(itertools.chain.from_iterable(paired_cols))

# train_preds_path = ou_eval_dir / f"{ou_name}_{method}_train_preds.csv"
# with open(train_preds_path, "w+") as train_preds_file:
#     temp = np.concatenate((X_train, y_train, y_train_pred), axis=1)
#     train_result_df = pd.DataFrame(temp, columns=feat_cols + target_cols +
#                                     [f"pred_{col}" for col in target_cols])
#     train_result_df[reordered_cols].to_csv(train_preds_file, float_format="%.1f", index=False)

# test_preds_path = ou_eval_dir / f"{ou_name}_{method}_test_preds.csv"
# with open(test_preds_path, "w+") as test_preds_file:
#     temp = np.concatenate((X_test, y_test, y_test_pred), axis=1)
#     test_result_df = pd.DataFrame(temp, columns=feat_cols + target_cols +
#                                     [f"pred_{col}" for col in target_cols])
#     test_result_df[reordered_cols].to_csv(test_preds_file, float_format="%.1f", index=False)


# if __name__ == "__main__":

#     ou_eval_path = ou_eval_dir / f"{ou_name}_{method}_summary.txt"
#     with open(ou_eval_path, "w+") as eval_file:
#         eval_file.write(f"\n============= Model Summary for {ou_name} Model: {method} =============\n")
#         eval_file.write(f"Num Runs used: {ou_name_to_nruns[ou_name]}\n")
#         eval_file.write(f"Features used: {feat_cols}\n")
#         eval_file.write(f"Num Features used: {len(feat_cols)}\n")
#         eval_file.write(f"Targets estimated: {target_cols}\n")

#         for target_idx, target in enumerate(target_cols):
#             eval_file.write(f"===== Target: {target} =====\n")
#             train_target_pred = y_train_pred[:, target_idx]
#             train_target_true = y_train[:, target_idx]
#             mse = mean_squared_error(train_target_true, train_target_pred)
#             mae = mean_absolute_error(train_target_true, train_target_pred)
#             r2 = r2_score(train_target_true, train_target_pred)
#             eval_file.write(f"Train MSE: {round(mse, 2)}\n")
#             eval_file.write(f"Train MAE: {round(mae, 2)}\n")
#             eval_file.write(f"Train R^2: {round(r2, 2)}\n")

#             test_target_pred = y_test_pred[:, target_idx]
#             test_target_true = y_test[:, target_idx]
#             mse = mean_squared_error(test_target_true, test_target_pred)
#             mae = mean_absolute_error(test_target_true, test_target_pred)
#             r2 = r2_score(test_target_true, test_target_pred)
#             eval_file.write(f"Test MSE: {round(mse, 2)}\n")
#             eval_file.write(f"Test MAE: {round(mae, 2)}\n")
#             eval_file.write(f"Test R^2: {round(r2, 2)}\n")
#         eval_file.write("======================== END SUMMARY ========================\n")
