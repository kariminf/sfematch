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

import numpy as np

def dcg_at_k(relevances, k):
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(ranked_ids, relevant_set, k):
    relevances = [1.0 if rid in relevant_set else 0.0 for rid in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_ids, relevant_set):
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank

    return 0.0

def r_p_at_k(ranked_ids, relevant_set, k):
    """Calculates Recall@k, Precision@k"""
    top_k = ranked_ids[:k]
    n_hit = sum(1 for rid in top_k if rid in relevant_set)
    return n_hit / len(relevant_set), n_hit / k


def cosine_sim_matrix(A, B):
    """A: (n_a, d), B: (n_b, d). Returns (n_a, n_b) cosine similarity, NaN-safe
    (rows with any NaN produce NaN similarity, excluded downstream)."""
    A_valid = ~np.isnan(A).any(axis=1)
    B_valid = ~np.isnan(B).any(axis=1)
    A0 = np.nan_to_num(A)
    B0 = np.nan_to_num(B)
    A_norm = A0 / (np.linalg.norm(A0, axis=1, keepdims=True) + 1e-12)
    B_norm = B0 / (np.linalg.norm(B0, axis=1, keepdims=True) + 1e-12)
    sim = A_norm @ B_norm.T
    sim[~A_valid, :] = np.nan
    sim[:, ~B_valid] = np.nan
    return sim