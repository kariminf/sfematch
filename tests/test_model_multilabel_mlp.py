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

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.multilabel_mlp import train_multilabel_model, predict_proba, save_model, load_model



if __name__ == "__main__":
    # smoke test with synthetic data
    np.random.seed(0)
    n, hidden, n_labels = 2000, 64, 5
    X = np.random.randn(n, hidden).astype(np.float32)
    Y = (np.random.rand(n, n_labels) > 0.7).astype(np.float32)

    n_train = int(n * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    model = train_multilabel_model(X_train, Y_train, X_val, Y_val, epochs=3)
    proba = predict_proba(model, X_val[:5])
    print("proba shape:", proba.shape)
    print(proba)

    save_model(model, "/tmp/multilabel_mlp.pt")
    loaded = load_model("/tmp/multilabel_mlp.pt")
    proba2 = predict_proba(loaded, X_val[:5])
    assert np.allclose(proba, proba2, atol=1e-5)
    print("save/load roundtrip OK")