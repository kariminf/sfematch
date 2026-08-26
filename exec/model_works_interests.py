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
import csv
import sys
import os
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.datasets import load_embeddings, load_tsv
from sfematch.model.multilabel_mlp import load_model, predict_proba



# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------

def read_both(tsv_path, emb_path):
    emb = load_embeddings(emb_path)
    df = load_tsv(tsv_path)

    return df["id"].to_numpy().astype(object), emb


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def process_split(name, ids, emb, model, device, out_dir: Path):
    print(f"\n=== {name} ===")
    print(f"{len(ids)} rows to encode")

    os.makedirs(out_dir, exist_ok=True)

    probs = predict_proba(model, emb, device=device)
    probs = probs.astype(np.float32)

    np.save(out_dir / f"{name}_ids.npy", ids)
    np.save(out_dir / f"{name}_probs.npy", probs)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate probabilistic fields models for experts based on interests and works."
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    interest_ids, interest_emb = read_both(config_json["interests_tsv"], config_json["interests_emb"])
    work_ids, work_emb = read_both(config_json["works_tsv"], config_json["works_emb"])

    model = load_model(config_json["model"])

    name = config_json["name"]
    out_path = Path(config_json["output"])

    process_split(f"{name}_interests", interest_ids, interest_emb, model, device, out_path / "interests")
    process_split(f"{name}_works", work_ids, work_emb, model, device, out_path / "works")

    print("\ndone.")


if __name__ == "__main__":
    main()
