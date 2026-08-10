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
Read experts_info.json + experts_labels.json and compute, for each label set
(Arxiv, ACM-CCS-L1, CCSF) and each employment-status group (Faculty, Associate,
PhD):

    - minority class count
    - majority class count
    - avg samples per class
    - median samples per class
    - number of experts with 1 / 2 / 3 / 4 active labels

Output: a text report + a filled-in LaTeX table matching the requested layout.

--- Expected input formats -------------------------------------------------

experts_info.json:
{
    "XPRT0001": {
        "given_names": ["Amar"],
        "family_name": "Balla",
        "employment_status": "Faculty",
        ...
    },
    ...
}

experts_labels.json:
{
    "XPRT0001": {
        "Arxiv": [0, 5, 8, 11, 13, 38],   # indices of active labels
        "CCS-L1": [1, 5, 6, 10],
        "CCSF": [0, 2]
    },
    ...
}

Each list gives the indices (0-based) of the labels that are "on" (i.e. the
positions that would be 1 in a multi-hot vector) for that expert.

-----------------------------------------------------------------------------

Configuration you may need to tweak (see CONFIG section below):
- LABEL_SET_SIZES: total number of possible labels in each set (needed
  because a label that is never chosen by anyone would otherwise be
  invisible if we only looked at the data).
- STATUS_GROUPS: how raw "employment_status" strings map to the three
  columns (Faculty / Associate / PhD).

Usage:
    python expert_label_stats.py experts_info.json experts_labels.json output.txt
"""

import sys
import json
import argparse
import statistics
from collections import defaultdict, Counter


# ============================= CONFIG =======================================

# Display name -> (json key in experts_labels.json, total number of labels)
LABEL_SETS = {
    "Arxiv":       {"key": "Arxiv",   "size": 40},
    "ACM-CCS-L1":  {"key": "CCS-L1",  "size": 13},
    "CCSF":        {"key": "CCSF",    "size": 6},
}

# Order of columns in the output table
STATUS_COLUMNS = ["Faculty", "Associate", "PhD"]


def normalize_status(raw_status):
    """
    Map a raw employment_status string from experts_info.json to one of
    STATUS_COLUMNS, or None if it doesn't belong to any of them.
    Adjust the matching rules below if your raw values differ
    (e.g. "Assistant Professor", "Associate Professor", "PhD Student", ...).
    """
    if not raw_status:
        return None
    s = raw_status.strip().lower()
    if "faculty" in s:
        return "Faculty"
    if "associat" in s:
        return "Associate"
    if "phd" in s or "ph.d" in s:
        return "PhD"
    return None

# =============================================================================


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_expert_status(experts_info):
    """expert_id -> normalized status (or None)"""
    status_of = {}
    unmapped = Counter()
    for expert_id, info in experts_info.items():
        raw = info.get("employment_status")
        status = normalize_status(raw)
        status_of[expert_id] = status
        if status is None:
            unmapped[raw] += 1
    return status_of, unmapped


def compute_group_stats(expert_ids, labels_by_expert, label_key, num_labels):
    """
    expert_ids: iterable of expert ids belonging to one status group
    labels_by_expert: dict expert_id -> {label_set_key: [indices]}
    label_key: which label set to look at (e.g. "Arxiv")
    num_labels: total number of labels in that set

    Returns a dict of stats, or None if the group is empty.
    """
    per_class_counts = [0] * num_labels
    labels_per_expert_hist = Counter()
    n_experts_considered = 0

    for eid in expert_ids:
        expert_labels = labels_by_expert.get(eid, {})
        active = expert_labels.get(label_key, [])
        active = [i for i in active if 0 <= i < num_labels]

        for i in active:
            per_class_counts[i] += 1

        labels_per_expert_hist[len(active)] += 1
        n_experts_considered += 1

    if n_experts_considered == 0:
        return None

    return {
        "n_experts": n_experts_considered,
        "minority_class_count": min(per_class_counts),
        "majority_class_count": max(per_class_counts),
        "avg_samples_per_class": round(statistics.mean(per_class_counts), 2),
        "median_samples_per_class": statistics.median(per_class_counts),
        "n_with_1_label": labels_per_expert_hist.get(1, 0),
        "n_with_2_labels": labels_per_expert_hist.get(2, 0),
        "n_with_3_labels": labels_per_expert_hist.get(3, 0),
        "n_with_4_labels": labels_per_expert_hist.get(4, 0),
        "per_class_counts": per_class_counts,
    }


def build_report(all_stats):
    lines = []
    for set_name, per_status in all_stats.items():
        lines.append(f"=== {set_name} ===")
        for status in STATUS_COLUMNS:
            stats = per_status.get(status)
            lines.append(f"-- {status} --")
            if stats is None:
                lines.append("  (no experts in this group)")
                continue
            lines.append(f"  n_experts: {stats['n_experts']}")
            lines.append(f"  minority_class_count: {stats['minority_class_count']}")
            lines.append(f"  majority_class_count: {stats['majority_class_count']}")
            lines.append(f"  avg_samples_per_class: {stats['avg_samples_per_class']}")
            lines.append(f"  median_samples_per_class: {stats['median_samples_per_class']}")
            lines.append(f"  n_with_1_label: {stats['n_with_1_label']}")
            lines.append(f"  n_with_2_labels: {stats['n_with_2_labels']}")
            lines.append(f"  n_with_3_labels: {stats['n_with_3_labels']}")
            lines.append(f"  n_with_4_labels: {stats['n_with_4_labels']}")
        lines.append("")
    return "\n".join(lines)


def build_latex_table(all_stats):
    def cell(set_name, status, field):
        stats = all_stats[set_name].get(status)
        if stats is None:
            return "--"
        return str(stats[field])

    fields = [
        ("minority class count", "minority_class_count"),
        ("majority class count", "majority_class_count"),
        ("avg samples per class", "avg_samples_per_class"),
        ("median samples per class", "median_samples_per_class"),
        ("number of experts with 1 label", "n_with_1_label"),
        ("number of experts with 2 labels", "n_with_2_labels"),
        ("number of experts with 3 labels", "n_with_3_labels"),
        ("number of experts with 4 labels", "n_with_4_labels"),
    ]

    set_names = list(LABEL_SETS.keys())
    sizes = {name: LABEL_SETS[name]["size"] for name in set_names}

    lines = []
    header_top = "  & " + " & ".join(
        f"\\multicolumn{{3}}{{l}}{{{name} ({sizes[name]} labels)}}" for name in set_names
    ) + " \\\\"
    header_bottom = "\t\tCriterion & " + " & ".join(
        " & ".join(STATUS_COLUMNS) for _ in set_names
    ) + " \\\\"

    lines.append(header_top)
    lines.append(header_bottom)
    lines.append("\t\t\\hline")

    for label, field in fields:
        row_cells = []
        for set_name in set_names:
            for status in STATUS_COLUMNS:
                row_cells.append(cell(set_name, status, field))
        lines.append(f"\t\t{label} & " + " & ".join(row_cells) + " \\\\")

    lines.append("\t\t\\hline")
    return "\n".join(lines)
