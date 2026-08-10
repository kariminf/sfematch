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
arxiv_tsv_stats.py

Computes statistics on the TSV file produced by arxiv_json_to_tsv.py:
in particular, how many papers have exactly 1 label, 2 labels, 3 labels,
etc. (a paper's "labels" = its binary field/category columns that are 1).

Also reports some extra useful stats along the way: per-field totals,
mean/median labels per paper, and papers with 0 labels (i.e. none of the
chosen fields matched -- can happen if a paper is cross-listed entirely
outside the domain you extracted the binary columns for).

Assumes the TSV has this structure (as produced by arxiv_json_to_tsv.py):

    id \t title \t abstract \t keywords \t field1 \t field2 \t ... \t fieldN

Where field1..fieldN are 0/1 columns. Everything before them (id, title,
abstract, keywords) is treated as non-label metadata and ignored for the
counting -- field columns are auto-detected as "everything after keywords",
or you can pass an explicit list.

Usage as a library
-------------------
    from arxiv_tsv_stats import label_statistics

    stats = label_statistics("cs_papers.tsv")
    print(stats["label_count_distribution"])
    print(stats["per_field_counts"])

Usage as a script
-------------------
    python arxiv_tsv_stats.py --input cs_papers.tsv
    python arxiv_tsv_stats.py --input cs_papers.tsv --output cs_stats.txt
    python arxiv_tsv_stats.py --input cs_papers.tsv --fields cs.LG,cs.CV,cs.CL
"""

import argparse
import sys

import pandas as pd


NON_FIELD_COLUMNS = ["id", "title", "abstract", "keywords"]


def label_statistics(
    tsv_path,
    fields=None,
    output=None,
    chunksize=200_000,
):
    """
    Compute label-count statistics from a TSV file produced by
    arxiv_json_to_tsv.py.

    Parameters
    ----------
    tsv_path : str
        Path to the input .tsv file.
    fields : list[str] or None
        Which columns to treat as binary label columns. If None,
        auto-detected as every column after "keywords" in the header.
    output : str or None
        If given, write a human-readable text summary to this path.
    chunksize : int
        Rows per chunk when streaming the file with pandas, to keep
        memory flat on very large TSVs. Set to None to read the whole
        file at once instead.

    Returns
    -------
    dict with keys:
        "n_papers"                    : total number of papers
        "label_count_distribution"    : pandas Series, index = number of
                                         labels, value = number of papers
                                         with that many labels
        "per_field_counts"            : pandas Series, index = field name,
                                         value = number of papers tagged
                                         with that field
        "mean_labels_per_paper"       : float
        "median_labels_per_paper"     : float
        "n_papers_zero_labels"        : int
    """
    # Peek at header to determine field columns if not given
    header = pd.read_csv(tsv_path, sep="\t", quoting=3, nrows=0).columns.tolist()

    if fields is None:
        fields = [c for c in header if c not in NON_FIELD_COLUMNS]

    missing = [f for f in fields if f not in header]
    if missing:
        raise ValueError(f"These fields are not columns in the TSV: {missing}")

    usecols = fields  # we only need the binary columns for these stats

    label_count_dist = pd.Series(dtype="int64")
    per_field_counts = pd.Series(0, index=fields, dtype="int64")
    n_papers = 0
    sum_labels = 0
    n_bad_values = 0

    # NOTE: dtype is deliberately NOT forced to int8 here. Ragged/malformed
    # rows (e.g. a stray literal tab that slipped through cleaning, or a
    # truncated line) can produce missing values in these columns, and
    # forcing an integer dtype upfront makes pandas raise on that. Instead
    # we read as float (which tolerates NaN), then clean up afterward.
    reader = pd.read_csv(
        tsv_path, sep="\t", quoting=3, usecols=usecols, dtype="float64",
        chunksize=chunksize,
    )

    for chunk in reader:
        n_missing = int(chunk[fields].isna().sum().sum())
        if n_missing:
            n_bad_values += n_missing
            chunk[fields] = chunk[fields].fillna(0)

        chunk[fields] = chunk[fields].astype("int8")

        labels_per_row = chunk[fields].sum(axis=1)

        n_papers += len(chunk)
        sum_labels += int(labels_per_row.sum())

        chunk_dist = labels_per_row.value_counts()
        label_count_dist = label_count_dist.add(chunk_dist, fill_value=0)

        per_field_counts = per_field_counts.add(chunk[fields].sum(), fill_value=0)

    label_count_dist = label_count_dist.sort_index().astype("int64")
    per_field_counts = per_field_counts.sort_values(ascending=False).astype("int64")

    mean_labels = sum_labels / n_papers if n_papers else 0.0

    # median needs the raw distribution, computable from label_count_dist
    if n_papers:
        expanded_index = label_count_dist.index.repeat(label_count_dist.values)
        median_labels = float(pd.Series(expanded_index).median())
    else:
        median_labels = 0.0

    n_zero_labels = int(label_count_dist.get(0, 0))

    if n_bad_values:
        print(
            f"Warning: {n_bad_values:,} missing/malformed values found in "
            f"label columns and treated as 0. This usually means some rows "
            f"in the TSV are ragged (e.g. a stray tab/newline slipped into "
            f"a title or abstract). Consider re-running "
            f"arxiv_json_to_tsv.py if this number looks large.",
            file=sys.stderr,
        )

    stats = {
        "n_papers": n_papers,
        "label_count_distribution": label_count_dist,
        "per_field_counts": per_field_counts,
        "mean_labels_per_paper": mean_labels,
        "median_labels_per_paper": median_labels,
        "n_papers_zero_labels": n_zero_labels,
        "n_bad_values": n_bad_values,
    }

    summary_text = _format_summary(stats)
    print(summary_text)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"\nSummary written to {output}", file=sys.stderr)

    return stats


def _format_summary(stats):
    lines = []
    lines.append(f"Total papers: {stats['n_papers']:,}")
    lines.append(f"Mean labels per paper: {stats['mean_labels_per_paper']:.3f}")
    lines.append(f"Median labels per paper: {stats['median_labels_per_paper']:.1f}")
    lines.append(f"Papers with 0 labels: {stats['n_papers_zero_labels']:,}")
    lines.append("")
    lines.append("Label count distribution:")
    lines.append(f"{'# labels':>10} | {'# papers':>10} | {'% of total':>10}")
    lines.append("-" * 36)
    total = stats["n_papers"] or 1
    for n_labels, n_papers in stats["label_count_distribution"].items():
        pct = 100 * n_papers / total
        lines.append(f"{n_labels:>10} | {n_papers:>10,} | {pct:>9.2f}%")
    lines.append("")
    lines.append("Per-field paper counts (top 20):")
    lines.append(f"{'field':>10} | {'# papers':>10}")
    lines.append("-" * 25)
    for field, count in stats["per_field_counts"].head(20).items():
        lines.append(f"{field:>10} | {count:>10,}")

    return "\n".join(lines)
