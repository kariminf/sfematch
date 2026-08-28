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

def rank(S, E, s_ids, e_ids, sim_fn, top_k=None):
    assert S.shape[0] == len(s_ids), "S and s_ids length mismatch"
    assert E.shape[0] == len(e_ids), "E and e_ids length mismatch"

    sim = sim_fn(S, E)

    sub_exp = {}
    for i, s_id in enumerate(s_ids):
        row = sim[i]
        valid = ~np.isnan(row)
        valid_idx = np.where(valid)[0]

        order = valid_idx[np.argsort(-row[valid_idx])]

        if top_k is not None:
            order = order[:top_k]

        sub_exp[s_id] = {
            "ranked_ids": [e_ids[j] for j in order],
            "ranked_sims": [float(row[j]) for j in order],
        }

    return sub_exp
