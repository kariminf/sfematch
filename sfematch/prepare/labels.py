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

import pandas as pd

# ---------------------------------------------------------------------------
# Mapping 1: arXiv cs.* -> ACM CCS 2012, Level 1
# ---------------------------------------------------------------------------

ACM_CCS2012L1 = [
    "Applied computing",
    "Computer systems organization",
    "Computing methodologies",
    "General and reference",
    "Hardware",
    "Human-centered computing",
    "Information systems",
    "Mathematics of computing",
    "Networks",
    "Security and privacy",
    "Social and professional topics",
    "Software and its engineering",
    "Theory of computation",
]

ARXIV_CS_TO_CCS2012L1 = {
    "cs.AI": [2],
    "cs.AR": [1],
    "cs.CC": [12],
    "cs.CE": [0],
    "cs.CG": [12],
    "cs.CL": [2],
    "cs.CR": [9],
    "cs.CV": [2],
    "cs.CY": [10],
    "cs.DB": [6],
    "cs.DC": [1],
    "cs.DL": [6],
    "cs.DM": [7],
    "cs.DS": [12],
    "cs.ET": [0, 1, 4],
    "cs.FL": [12],
    "cs.GL": [3],
    "cs.GR": [2],
    "cs.GT": [12],
    "cs.HC": [5],
    "cs.IR": [6],
    "cs.IT": [7],
    "cs.LG": [2],
    "cs.LO": [12],
    "cs.MA": [2],
    "cs.MM": [5, 6],
    "cs.MS": [7],
    "cs.NA": [7],
    "cs.NE": [2],
    "cs.NI": [8],
    "cs.OH": [3],
    "cs.OS": [1],
    "cs.PF": [1],
    "cs.PL": [11],
    "cs.RO": [2],
    "cs.SC": [7, 11],
    "cs.SD": [2],
    "cs.SE": [11],
    "cs.SI": [6],
    "cs.SY": [1],
}


# ---------------------------------------------------------------------------
# Mapping 2: arXiv cs.* -> Core Computer Science Fields
# ---------------------------------------------------------------------------

CCSF = [
    "Artificial Intelligence & Data",
    "Core Foundations",
    "Information Systems",
    "Security & Privacy",
    "Software & Programming",
    "Systems & Infrastructure",
]

ARXIV_CS_TO_CCSF = {
    "cs.AI": [0],
    "cs.AR": [5],
    "cs.CC": [1],
    "cs.CE": [5],
    "cs.CG": [1],
    "cs.CL": [0],
    "cs.CR": [3],
    "cs.CV": [0],
    "cs.CY": [2],
    "cs.DB": [0, 2],
    "cs.DC": [4, 5],
    "cs.DL": [2],
    "cs.DM": [1],
    "cs.DS": [1],
    "cs.ET": [0, 5],
    "cs.FL": [1],
    "cs.GL": [1],
    "cs.GR": [1, 4],
    "cs.GT": [1],
    "cs.HC": [2, 4],
    "cs.IR": [0, 2],
    "cs.IT": [1],
    "cs.LG": [0],
    "cs.LO": [1],
    "cs.MA": [0, 1],
    "cs.MM": [0, 2],
    "cs.MS": [1],
    "cs.NA": [1],
    "cs.NE": [0],
    "cs.NI": [5],
    "cs.OH": [1],
    "cs.OS": [5],
    "cs.PF": [5],
    "cs.PL": [4],
    "cs.RO": [0, 5],
    "cs.SC": [1, 4],
    "cs.SD": [0],
    "cs.SE": [4],
    "cs.SI": [2],
    "cs.SY": [5],
}


# ---------------------------------------------------------------------------
# Core mapping function
# ---------------------------------------------------------------------------

def map_labels(labels_tsv_path, target_names, mapping, output=None):
    """
    Read a labels TSV (id + source 0/1 fields) and produce a new labels
    TSV (id + target 0/1 fields), where each target field is 1 for a
    paper if ANY of its positive source fields maps to that target
    (fuzzy / multi-target mapping, logical OR).

    Parameters
    ----------
    labels_tsv_path : str
        Path to the input labels TSV. Must have an "id" column plus one
        column per source field (e.g. cs.AI, cs.LG, ...).
    target_names : list[str]
        Ordered list of target category names (columns in the output).
    mapping : dict[str, list[int]]
        Maps each source field name to a list of indices into
        `target_names` it should contribute to.
    output : str or None
        Output path. Default: "<input>_mapped.tsv".

    Returns
    -------
    df_out : pandas DataFrame with columns ["id"] + target_names
    """
    df = pd.read_csv(
        labels_tsv_path, sep="\t", quoting=3, keep_default_na=False, na_values=[]
    )

    if "id" not in df.columns:
        raise ValueError("Input labels TSV has no 'id' column, which is required.")

    source_fields = [c for c in df.columns if c != "id"]

    unmapped = [f for f in source_fields if f not in mapping]
    if unmapped:
        print(
            f"Warning: these source fields have no entry in the mapping "
            f"and will be ignored: {unmapped}",
            file=sys.stderr,
        )

    n = len(df)
    target_matrix = pd.DataFrame(
        0, index=df.index, columns=target_names, dtype="int8"
    )

    for src_field in source_fields:
        if src_field not in mapping:
            continue
        src_positive = df[src_field].astype(int) == 1
        for target_idx in mapping[src_field]:
            target_col = target_names[target_idx]
            target_matrix[target_col] = (
                target_matrix[target_col] | src_positive.astype("int8")
            )

    df_out = pd.concat([df[["id"]], target_matrix], axis=1)

    if output is None:
        output = labels_tsv_path.rsplit(".", 1)[0] + "_mapped.tsv"

    df_out.to_csv(output, sep="\t", index=False, quoting=3)
    print(f"Wrote {len(df_out):,} rows to {output}", file=sys.stderr)

    return df_out


def generate_all_mappings(
    labels_tsv_path,
    output_ccs2012l1=None,
    output_ccsf=None,
):
    """
    Convenience wrapper: generate BOTH mapped label files (ACM CCS2012 L1
    and Core CS Fields) from a single source labels TSV.

    Parameters
    ----------
    labels_tsv_path : str
        Path to the input labels TSV.
    output_ccs2012l1 : str or None
        Output path for the CCS2012 L1 labels file. Default:
        "<input>_ccs2012l1.tsv".
    output_ccsf : str or None
        Output path for the Core CS Fields labels file. Default:
        "<input>_ccsf.tsv".

    Returns
    -------
    (df_ccs2012l1, df_ccsf) : tuple of pandas DataFrames
    """
    if output_ccs2012l1 is None:
        output_ccs2012l1 = labels_tsv_path.rsplit(".", 1)[0] + "_ccs2012l1.tsv"
    if output_ccsf is None:
        output_ccsf = labels_tsv_path.rsplit(".", 1)[0] + "_ccsf.tsv"

    df_ccs2012l1 = map_labels(
        labels_tsv_path, ACM_CCS2012L1, ARXIV_CS_TO_CCS2012L1, output_ccs2012l1
    )
    df_ccsf = map_labels(
        labels_tsv_path, CCSF, ARXIV_CS_TO_CCSF, output_ccsf
    )

    return df_ccs2012l1, df_ccsf
