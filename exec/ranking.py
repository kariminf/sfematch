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

from sfematch.match.rank import rank
from sfematch.match.similarity_score import cosine_similarity, euclidean_similarity_bounded, mae_similarity


SIM_FNS = {
    "cosine": cosine_similarity,
    "euclean": euclidean_similarity_bounded,
    "mae":  mae_similarity
}

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_ids_probs(ids_path, probs_path):

    ids = np.load(ids_path, allow_pickle=True)
    probs = np.load(probs_path)

    return ids, probs

def validate_top_k(top_k):
    if top_k is None:
        return None
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"top_k must be None or an int, got {type(top_k).__name__}: {top_k!r}")
    if top_k <= 0:
        raise ValueError(f"top_k must be > 0, got {top_k}")
    return top_k

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
            description="Match experts to subjects."
        )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    print("loading models ...")
    s_ids, S = load_ids_probs(config_json["subject_ids"], config_json["subject_probs"])
    e_ids, E = load_ids_probs(config_json["expert_ids"], config_json["expert_probs"])

    

    sim_id = "cosine"
    if config_json["sim"] in SIM_FNS:
        sim_id = config_json["sim"]

    sim_fn = SIM_FNS[sim_id]

    top_k = validate_top_k(config_json.get("top_k"))

    print(f"Matching using {sim_id} similarity and top_k = {top_k} ...")

    ranks_dict = rank(S, E, s_ids, e_ids, sim_fn, top_k)

    out_dir = config_json["output_match"]

    with open(out_dir, "w", encoding="utf-8") as f:
        json.dump(ranks_dict, f, indent=4, ensure_ascii=True)

    
    print(f"\ndone. outputs in {out_dir}")


if __name__ == "__main__":
    main()
