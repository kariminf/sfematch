#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2026 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2026	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import numpy as np
import os
import sys

from sklearn.metrics import classification_report, mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.datasets       import load_embeddings, load_multilabel_labels
from sfematch.model.multilabel_mlp import  load_model, predict_proba





def compute_metrics(Y_true, proba, labels, threshold=0.5):
    Y_pred = (proba >= threshold).astype(int)

    # print(np.unique(Y_true), np.unique(Y_pred))
    report = classification_report(
        Y_true, Y_pred, zero_division=0, target_names=labels
    )
    mae = mean_absolute_error(Y_true, proba)  # proba vs binary ground truth, elementwise

    result = f"Mean Absolute Error = {mae}\n\n Classification report:\n"
    result += str(report)

    return result


def load_datasets(x_url: str, y_url: str):

    emb = load_embeddings(x_url)
    y_df, labels = load_multilabel_labels(y_url)

    if emb.shape[0] != len(y_df):
        raise ValueError(f"embeddings ({emb.shape[0]}) and labels ({len(y_df)}) have different sizes")

    X = np.asarray(emb).astype(np.float32)
    Y = y_df[labels].to_numpy().astype(int)
    return X, Y, labels
    


def main():
    parser = argparse.ArgumentParser(
        description="Test an MLP with one hidden layer for classification."
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    model = load_model(config_json["model"])


    X, Y, labels = load_datasets(config_json["x_test"], config_json["y_test"])

    H = predict_proba(model, X)

    # print(f"Y: {Y.shape}, H: {H.shape}")

    report = compute_metrics(Y, H, labels, threshold=config_json["threshold"])

    with open(config_json["out_report"], "w", encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()