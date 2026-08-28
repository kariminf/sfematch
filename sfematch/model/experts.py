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


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def pool_expert_vector(item_ids, id_to_index, probs, pool):
    """Gathers the prob rows for item_ids present in id_to_index and pools them.
    Returns None if none of item_ids were found (e.g. expert has no interests)."""
    rows = [probs[id_to_index[i]] for i in item_ids if i in id_to_index]
    if not rows:
        return None
    stacked = np.stack(rows, axis=0)
    if pool == "avg":
        return stacked.mean(axis=0)
    if pool == "max":
        return stacked.max(axis=0)
    raise ValueError(f"unknown pool: {pool}")


def build_expert_models(expert_item_ids, id_to_index, name_probs, pool):
    """For every taxonomy, pools each expert's items into one vector.
    Returns {tax: {expert_id: vector}} (experts with nothing found are omitted)."""
    models = {}
    for name, probs in name_probs.items():
        expert_vecs = {}
        for expert, item_ids in expert_item_ids.items():
            vec = pool_expert_vector(item_ids, id_to_index, probs, pool)
            if vec is not None:
                expert_vecs[expert] = vec
        models[name] = expert_vecs
    return models