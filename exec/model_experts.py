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

"""
model_experts.py

Builds per-expert models by pooling the taxonomy classifier outputs produced by
model_works_interests.py, then mixes the "interest" model I and the "work"
model W with several alpha weights.

Inputs
------
ExpertsModeling/experts_interests.json
    {expert_id: [interest_id, interest_id, ...], ...}

ExpertsModeling/experts_works.json
    {expert_id: [{"id": work_id, "pos": ..., "nbr": ...}, ...], ...}

FinalModels/interests/  and  FinalModels/works/
    Output of model_works_interests.py: ids.npy + {tax}_probs.npy
    for tax in {arxiv, ccs2012l1, ccsf}.

Per expert / per taxonomy
--------------------------
I = pool(probs of the expert's interests)   pool in {max, avg}
W = pool(probs of the expert's works)       pool in {max, avg}
Expert = alpha * W + (1 - alpha) * I        alpha in {0.0, 0.25, 0.5, 0.75, 1.0}

Output
------
FinalModels/experts/
    ids_{pool}_sbert_{tax}.npy               -> expert ids, row order for the file below
    mixed_{pool}_a{alpha}_sbert_{tax}.npy    -> (n_experts, n_labels) mixed model
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.datasets import load_embeddings, load_tsv
from sfematch.model.experts import pool_expert_vector, build_expert_models


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def load_probs_dir(dir_path: Path, t=None):

    name_probs = {}
    name = ""

    suff = f"_{t}_probs.npy" if t else "_probs.npy"

    for probs_file in dir_path.glob(f"*{suff}"):
        name = probs_file.name.removesuffix(f"{suff}")
        name_probs[name] = np.load(probs_file)

    # just one id suffice (all are similar so take the last name)
    id_f = f"{name}_{t}_ids.npy" if t else f"{name}_ids.npy"
    ids = np.load(dir_path / id_f, allow_pickle=True)
    id_to_index = {str(v): i for i, v in enumerate(ids)}

    return id_to_index, name_probs


def load_expert_interests(json_path: Path):
    """experts_interests.json: {expert_id: [interest_id, ...]}"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {expert: list(ids) for expert, ids in data.items()}


def load_expert_works(json_path: Path):
    """experts_works.json: {expert_id: [{"id": work_id, "pos": .., "nbr": ..}, ...]}"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {expert: [item["id"] for item in items] for expert, items in data.items()}

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
            description="Model experts."
        )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    print("loading pooled classifier outputs (interests, works)...")
    interest_id_to_index, interest_name_probs = load_probs_dir(Path(config_json["interests_dir"]), t="interests")
    work_id_to_index, work_name_probs = load_probs_dir(Path(config_json["works_dir"]), t="works")

    print("loading expert -> interests / works mappings...")
    expert_interest_ids = load_expert_interests(config_json["interests_json"])
    expert_work_ids = load_expert_works(config_json["works_json"])

    # "files are joined": union of experts seen in either mapping
    all_experts = sorted(set(expert_interest_ids) | set(expert_work_ids))
    print(f"{len(all_experts)} experts total")
    for e in all_experts:
        expert_interest_ids.setdefault(e, [])
        expert_work_ids.setdefault(e, [])

    out_dir = Path(config_json["experts_dir"])
    os.makedirs(out_dir, exist_ok=True)

    for pool in config_json["pool"]:
        print(f"\n=== pool = {pool} ===")
        interest_models = build_expert_models(
            expert_interest_ids, interest_id_to_index, interest_name_probs, pool
        )
        work_models = build_expert_models(
            expert_work_ids, work_id_to_index, work_name_probs, pool
        )

        for name in interest_models:
            if name not in work_models:
                print(f"  [skip] '{name}' unavailable for pool '{pool}'")
                continue

            i_vecs = interest_models[name]
            w_vecs = work_models[name]

            # only experts that have BOTH an interest model and a work model
            common_experts = sorted(set(i_vecs) & set(w_vecs))
            missing = sorted(set(all_experts) - set(common_experts))
            if missing:
                shown = missing[:5]
                print(f"  [warn] tax={name}: {len(missing)} expert(s) skipped "
                      f"(missing interest and/or work model): {shown}"
                      f"{' ...' if len(missing) > 5 else ''}")

            if not common_experts:
                print(f"  [skip] tax={name}: no experts with both models")
                continue

            I = np.stack([i_vecs[e] for e in common_experts], axis=0)
            W = np.stack([w_vecs[e] for e in common_experts], axis=0)

            # expert-id ordering for this (pool, tax) combo, so rows can be
            # mapped back for every alpha value saved below
            ids_path = out_dir / f"ids_{pool}_{name}.npy"
            np.save(ids_path, np.array(common_experts, dtype=object))

            for alpha in config_json["alpha"]:
                mixed = alpha * W + (1.0 - alpha) * I
                out_path = out_dir / f"mixed_{pool}_a{alpha}_{name}.npy"
                np.save(out_path, mixed.astype(np.float32))
                print(f"  saved {out_path.name}  shape={mixed.shape}")

    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    main()
