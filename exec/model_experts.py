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
from pathlib import Path

import numpy as np


ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
def load_probs_dir(dir_path: Path):
    """Loads ids.npy + {tax}_probs.npy from a model_works_interests.py output dir.
    Returns (id_to_index: dict[str, int], tax_probs: {tax: (n, n_labels) array})."""
    ids_path = dir_path / "ids.npy"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"missing {ids_path}; run model_works_interests.py first"
        )
    ids = np.load(ids_path, allow_pickle=True)
    id_to_index = {str(v): i for i, v in enumerate(ids)}

    tax_probs = {}
    for tax in TAXONOMIES:
        p = dir_path / f"{tax}_probs.npy"
        if p.exists():
            tax_probs[tax] = np.load(p)
        else:
            print(f"  [warn] {p} not found -- taxonomy '{tax}' unavailable for {dir_path}")
    return id_to_index, tax_probs


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


def main():
    parser = argparse.ArgumentParser(
            description="Model experts."
        )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    print("loading pooled classifier outputs (interests, works)...")
    interest_id_to_index, interest_tax_probs = load_probs_dir(args.interests_dir)
    work_id_to_index, work_tax_probs = load_probs_dir(args.works_dir)

    print("loading expert -> interests / works mappings...")
    expert_interest_ids = load_expert_interests(args.interests_json)
    expert_work_ids = load_expert_works(args.works_json)

    # "files are joined": union of experts seen in either mapping
    all_experts = sorted(set(expert_interest_ids) | set(expert_work_ids))
    print(f"{len(all_experts)} experts total")
    for e in all_experts:
        expert_interest_ids.setdefault(e, [])
        expert_work_ids.setdefault(e, [])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for pool in POOLS:
        print(f"\n=== pool = {pool} ===")
        interest_models = build_expert_models(
            expert_interest_ids, interest_id_to_index, interest_tax_probs, pool
        )
        work_models = build_expert_models(
            expert_work_ids, work_id_to_index, work_tax_probs, pool
        )

        for tax in TAXONOMIES:
            if tax not in interest_models or tax not in work_models:
                print(f"  [skip] taxonomy '{tax}' unavailable for pool '{pool}'")
                continue

            i_vecs = interest_models[tax]
            w_vecs = work_models[tax]

            # only experts that have BOTH an interest model and a work model
            common_experts = sorted(set(i_vecs) & set(w_vecs))
            missing = sorted(set(all_experts) - set(common_experts))
            if missing:
                shown = missing[:5]
                print(f"  [warn] tax={tax}: {len(missing)} expert(s) skipped "
                      f"(missing interest and/or work model): {shown}"
                      f"{' ...' if len(missing) > 5 else ''}")

            if not common_experts:
                print(f"  [skip] tax={tax}: no experts with both models")
                continue

            I = np.stack([i_vecs[e] for e in common_experts], axis=0)
            W = np.stack([w_vecs[e] for e in common_experts], axis=0)

            # expert-id ordering for this (pool, tax) combo, so rows can be
            # mapped back for every alpha value saved below
            ids_path = args.out_dir / f"ids_{pool}_sbert_{tax}.npy"
            np.save(ids_path, np.array(common_experts, dtype=object))

            for alpha in ALPHAS:
                mixed = alpha * W + (1.0 - alpha) * I
                out_path = args.out_dir / f"mixed_{pool}_a{alpha}_sbert_{tax}.npy"
                np.save(out_path, mixed.astype(np.float32))
                print(f"  saved {out_path.name}  shape={mixed.shape}")

    print(f"\ndone. outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
