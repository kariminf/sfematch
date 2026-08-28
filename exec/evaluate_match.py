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
import pandas as pd
import sys

from typing import List, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.eval.rank_eval import mrr, r_p_at_k, ndcg_at_k, cosine_sim_matrix 


ID_COL = "id"

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def cut(id_list: Dict[str, List[str]], k) -> Dict[str, List[str]]:
    return {i: id_list[i][:k] for i in id_list}

def load_labels_gold(path):
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_values=[])
    df = df.set_index(ID_COL)
    s_ids = df.index.tolist()
    Y = df.astype(float).to_numpy()
    return s_ids, Y

def build_manual_gold_ranking_all(s_paths, e_paths):
    assert len(s_paths) == len(e_paths), "paths must have the same number"

    sims = []
    s_ids_ref = []
    e_ids_ref = []
    for s_path, e_path in zip(s_paths, e_paths):
        s_ids, s_Y = load_labels_gold(s_path)
        if not s_ids_ref:
            s_ids_ref = s_ids
        else:
            assert s_ids == s_ids_ref, f"subject id order mismatch for file '{s_path}'"

        e_ids, e_Y = load_labels_gold(e_path)
        if not e_ids_ref:
            e_ids_ref = e_ids
        else:
            assert e_ids == e_ids_ref, f"experts id order mismatch for file '{e_path}'"

        sims.append(cosine_sim_matrix(s_Y, e_Y))
    avg_sim = np.nanmean(np.stack(sims, axis=0), axis=0)  # (n_fyp, n_experts)

    gold = {}
    for i, fid in enumerate(s_ids_ref):
        row = avg_sim[i]
        order = np.argsort(-np.nan_to_num(row, nan=-np.inf))
        gold[fid] = [e_ids_ref[j] for j in order if not np.isnan(row[j])]
    return gold, avg_sim


def load_jury_gold(path):
    df = pd.read_csv(path, sep="\t", quoting=3, dtype=str, keep_default_na=False, na_values=[])
    gold = {}
    for _, row in df.iterrows():
        gold[row[ID_COL]] = [row["chair"], row["reviewer"], row["examiner"]]
    return gold

def evaluate_ranking(ranked_ids, relevant_set, ks):
    if not relevant_set:
        return None
    out = {"mrr": mrr(ranked_ids, relevant_set)}

    for k in ks:
        r, p = r_p_at_k(ranked_ids, relevant_set, k)
        out[f"recall@{k}"] = r
        out[f"precision@{k}"] = p
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, relevant_set, k)
    return out


def evaluate_matching_for_profile(match_dict, gold_dict, ks):
    per_subj_results = []
    for fid in gold_dict:
        relevant = set(gold_dict.get(fid, []))
        if not relevant:
            continue
        ranked_ids = match_dict.get(fid, [])

        res = evaluate_ranking(ranked_ids, relevant, ks)
        if res is not None:
            per_subj_results.append(res)

    if not per_subj_results:
        return None
    keys = per_subj_results[0].keys()
    return {k: float(np.mean([r[k] for r in per_subj_results])) for k in keys}

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


    with open(config_json["match_path"], encoding="utf-8") as f:
        match_dict = json.load(f)

    match_dict = {i: match_dict[i]["ranked_ids"] for i in match_dict}

    jury_gold = load_jury_gold(config_json["gold_path"])

    manual_order, avg_sim = build_manual_gold_ranking_all(config_json["s_paths"], config_json["e_paths"])

    ks = config_json["top_k"]
    name = config_json["name"]

    results = {}
    for gold_size in config_json["gold_size"]:
        gold_dict = cut(manual_order, gold_size)

        result = evaluate_matching_for_profile(match_dict, gold_dict, ks)

        if result is not None:
            results[f"{name}_vs_manual_k{gold_size}"] = result

    results[f"{name}_vs_jury"] = evaluate_matching_for_profile(match_dict, jury_gold, ks)

    out_path = config_json["out_path"]

    with open(os.path.join(out_path, "matching_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    matching_df = pd.DataFrame(results).T.sort_values("mrr", ascending=False)
    matching_df.to_csv(os.path.join(out_path, "matching_summary.csv"))
        


if __name__ == "__main__":
    main()
