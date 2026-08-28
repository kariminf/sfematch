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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.datasets import load_embeddings, load_multilabel_labels
from sfematch.model.multilabel_mlp import TrainConfig, train_multilabel_model, save_model



def load_datasets(x_url: str, y_url: str):

    emb = load_embeddings(x_url)
    y_df, labels = load_multilabel_labels(y_url)

    if emb.shape[0] != len(y_df):
        raise ValueError(f"embeddings ({emb.shape[0]}) and labels ({len(y_df)}) have different sizes")

    X = np.asarray(emb).astype(np.float32)
    Y = y_df[labels].to_numpy().astype(int)
    return X, Y
    


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP with one hidden layer for classification."
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    X, Y = load_datasets(config_json["x_train"], config_json["y_train"])

    config = TrainConfig(model_name=config_json["model_name"])
    config.fill(config_json)

    model = train_multilabel_model(X, Y, config=config)

    save_model(model, os.path.join(config_json["out_dir"], f"{config.model_name}.pt"))


if __name__ == "__main__":
    main()