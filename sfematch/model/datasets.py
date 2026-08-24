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

import pandas as pd

def iter_titles_abstracts(tsv_path, batch_size=2000):
    reader = pd.read_csv(
        tsv_path, sep="\t", usecols=["id", "title", "abstract"],
        dtype=str, chunksize=batch_size, keep_default_na=False, quoting=3
    )
    for chunk in reader:
        titles = chunk["title"].fillna("").str.strip().tolist()
        abstracts = chunk["abstract"].fillna("").str.strip().tolist()
        yield titles, abstracts


def count_rows_safe(tsv_path, chunksize=100_000):
    total = 0
    for chunk in pd.read_csv(tsv_path, sep="\t", usecols=["id"], dtype=str, chunksize=chunksize, quoting=3):
        total += len(chunk)
    return total
