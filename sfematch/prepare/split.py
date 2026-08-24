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
import numpy as np
import pandas as pd


NON_FIELD_COLUMNS = ["id", "title", "abstract", "keywords"]


def split_text_labels(
    tsv_path,
    fields=None,
    output_papers=None,
    output_labels=None,
):
    """
    Split a papers TSV into a text file (id, title, abstract, keywords)
    and a labels file (id, field1..fieldN), joined by "id".

    Parameters
    ----------
    tsv_path : str
        Path to the input .tsv file.
    fields : list[str] or None
        Which columns are the binary label columns. If None, auto-detected
        as every column after "keywords".
    output_papers : str or None
        Output path for the text file. Default: "<input>_text.tsv".
    output_labels : str or None
        Output path for the labels file. Default: "<input>_labels.tsv".

    Returns
    -------
    (df_papers, df_labels) : pandas DataFrames
    """
    # quoting=3 (QUOTE_NONE) since the TSV isn't CSV-quoted, and
    # keep_default_na=False so text fields aren't corrupted by pandas
    # treating strings like "NA" as missing values.
    df = pd.read_csv(
        tsv_path, sep="\t", quoting=3, keep_default_na=False, na_values=[]
    )

    if fields is None:
        fields = [c for c in df.columns if c not in NON_FIELD_COLUMNS]

    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"These fields are not columns in the TSV: {missing}")

    text_cols = [c for c in ["id", "title", "abstract", "keywords"] if c in df.columns]
    if "id" not in text_cols:
        raise ValueError("Input TSV has no 'id' column, which is required.")

    df_papers = df[text_cols]
    df_labels = df[["id"] + fields]

    if output_papers is None:
        output_papers = tsv_path.rsplit(".", 1)[0] + "_text.tsv"
    if output_labels is None:
        output_labels = tsv_path.rsplit(".", 1)[0] + "_labels.tsv"

    df_papers.to_csv(output_papers, sep="\t", index=False, quoting=3)
    df_labels.to_csv(output_labels, sep="\t", index=False, quoting=3)

    print(f"Wrote {len(df_papers):,} rows to {output_papers}", file=sys.stderr)
    print(f"Wrote {len(df_labels):,} rows to {output_labels}", file=sys.stderr)

    return df_papers, df_labels


def _iterative_stratify_indices(y, test_size, random_state=None):
    """
    Core iterative stratification algorithm (Sechidis et al., 2011).

    Parameters
    ----------
    y : np.ndarray of shape (n_samples, n_labels), values 0/1
    test_size : float in (0, 1)
    random_state : int or None

    Returns
    -------
    train_idx, test_idx : np.ndarray of row indices
    """
    rng = np.random.default_rng(random_state)

    n_samples, n_labels = y.shape
    proportions = {"train": 1.0 - test_size, "test": test_size}

    # Desired total size per split
    desired_total = {k: v * n_samples for k, v in proportions.items()}
    # Desired size per split per label
    label_totals = y.sum(axis=0)  # positives per label, shape (n_labels,)
    desired_per_label = {
        k: v * label_totals.astype(float) for k, v in proportions.items()
    }

    # For each label, the set of (still unassigned) example indices with that label
    label_to_examples = [set(np.nonzero(y[:, l])[0].tolist()) for l in range(n_labels)]

    # Precompute each example's positive label list for fast updates
    example_labels = [np.nonzero(y[i])[0].tolist() for i in range(n_samples)]

    assigned = {}  # idx -> "train"/"test"
    unassigned = set(range(n_samples))

    while True:
        # Remaining count of unassigned examples per label
        remaining_counts = np.array(
            [len(label_to_examples[l]) for l in range(n_labels)]
        )
        candidate_labels = np.nonzero(remaining_counts > 0)[0]
        if len(candidate_labels) == 0:
            break

        # Pick the rarest label among remaining (ties broken randomly)
        min_count = remaining_counts[candidate_labels].min()
        rarest = candidate_labels[remaining_counts[candidate_labels] == min_count]
        l_min = int(rng.choice(rarest))

        examples_for_label = list(label_to_examples[l_min])
        rng.shuffle(examples_for_label)

        for idx in examples_for_label:
            if idx not in unassigned:
                continue  # may have been assigned via another label already

            # Choose split with the greatest remaining desired quota for l_min
            quotas = {k: desired_per_label[k][l_min] for k in proportions}
            max_quota = max(quotas.values())
            best = [k for k, v in quotas.items() if v == max_quota]

            if len(best) > 1:
                # Tie-break by overall remaining desired size
                totals = {k: desired_total[k] for k in best}
                max_total = max(totals.values())
                best = [k for k, v in totals.items() if v == max_total]

            split = best[0] if len(best) == 1 else rng.choice(best)

            # Assign
            assigned[idx] = split
            unassigned.discard(idx)
            desired_total[split] -= 1
            for l in example_labels[idx]:
                desired_per_label[split][l] -= 1
                label_to_examples[l].discard(idx)

    # Any leftover examples with ALL-ZERO label rows never appear in any
    # label_to_examples set, so distribute them by remaining capacity.
    leftover = list(unassigned)
    rng.shuffle(leftover)
    for idx in leftover:
        split = max(desired_total, key=desired_total.get)
        assigned[idx] = split
        desired_total[split] -= 1

    train_idx = np.array([i for i, s in assigned.items() if s == "train"], dtype=int)
    test_idx = np.array([i for i, s in assigned.items() if s == "test"], dtype=int)

    return train_idx, test_idx


def _label_report(df, fields, train_idx, test_idx):
    lines = []
    n_total, n_train, n_test = len(df), len(train_idx), len(test_idx)
    lines.append(f"Total: {n_total:,}  |  Train: {n_train:,} "
                f"({100*n_train/n_total:.1f}%)  |  Test: {n_test:,} "
                f"({100*n_test/n_total:.1f}%)")
    lines.append("")

    y = df[fields].values
    y_train = y[train_idx]
    y_test = y[test_idx]

    lines.append(f"{'field':>10} | {'overall %':>10} | {'train %':>10} | {'test %':>10}")
    lines.append("-" * 48)
    for j, field in enumerate(fields):
        overall_pct = 100 * y[:, j].mean()
        train_pct = 100 * y_train[:, j].mean() if n_train else 0.0
        test_pct = 100 * y_test[:, j].mean() if n_test else 0.0
        lines.append(f"{field:>10} | {overall_pct:>9.2f}% | {train_pct:>9.2f}% "
                    f"| {test_pct:>9.2f}%")

    lines.append("")
    lines.append("Label-count distribution (# of simultaneous labels per paper):")
    lines.append(f"{'# labels':>10} | {'overall %':>10} | {'train %':>10} | {'test %':>10}")
    lines.append("-" * 48)
    counts_all = y.sum(axis=1)
    counts_train = y_train.sum(axis=1) if n_train else np.array([])
    counts_test = y_test.sum(axis=1) if n_test else np.array([])
    for n_labels in sorted(set(counts_all.tolist())):
        overall_pct = 100 * (counts_all == n_labels).mean()
        train_pct = 100 * (counts_train == n_labels).mean() if n_train else 0.0
        test_pct = 100 * (counts_test == n_labels).mean() if n_test else 0.0
        lines.append(f"{n_labels:>10} | {overall_pct:>9.2f}% | {train_pct:>9.2f}% "
                    f"| {test_pct:>9.2f}%")

    return "\n".join(lines)


def multilabel_train_test_split(
    tsv_path,
    test_size=0.2,
    fields=None,
    output_train=None,
    output_test=None,
    random_state=42,
    report=True,
):
    """
    Split a TSV file (as produced by arxiv_json_to_tsv.py) into train/test
    sets that are representative of the multi-label distribution.

    Parameters
    ----------
    tsv_path : str
        Path to the input .tsv file.
    test_size : float
        Fraction of examples to put in the test set. Default: 0.2 (20%).
    fields : list[str] or None
        Which columns are the binary label columns. If None, auto-detected
        as every column after "keywords".
    output_train, output_test : str or None
        Output paths. Default: "<input>_train.tsv" / "<input>_test.tsv".
    random_state : int or None
        Seed for reproducibility.
    report : bool
        If True, print a before/after per-label proportion comparison so
        you can see how representative the split turned out to be.

    Returns
    -------
    (df_train, df_test) : pandas DataFrames
    """
    # quoting=3 (QUOTE_NONE) since the TSV isn't CSV-quoted, and
    # keep_default_na=False so text fields aren't corrupted by pandas
    # treating strings like "NA" as missing values.
    df = pd.read_csv(
        tsv_path, sep="\t", quoting=3, keep_default_na=False, na_values=[]
    )

    if fields is None:
        fields = [c for c in df.columns if c not in NON_FIELD_COLUMNS]

    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise ValueError(f"These fields are not columns in the TSV: {missing}")

    y = df[fields].astype(int).values

    train_idx, test_idx = _iterative_stratify_indices(
        y, test_size=test_size, random_state=random_state
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    if output_train is None:
        output_train = tsv_path.rsplit(".", 1)[0] + "_train.tsv"
    if output_test is None:
        output_test = tsv_path.rsplit(".", 1)[0] + "_test.tsv"

    df_train.to_csv(output_train, sep="\t", index=False, quoting=3)
    df_test.to_csv(output_test, sep="\t", index=False, quoting=3)

    print(f"Wrote {len(df_train):,} rows to {output_train}", file=sys.stderr)
    print(f"Wrote {len(df_test):,} rows to {output_test}", file=sys.stderr)

    if report:
        print(_label_report(df, fields, train_idx, test_idx))

    return df_train, df_test
