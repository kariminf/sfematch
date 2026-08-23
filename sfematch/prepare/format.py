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

import json
import os
import re
import sys

from .arxiv_fields import DOMAIN_FIELDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[\n\t\r]+")


def clean_text(text):
    """
    Replace newlines/tabs/carriage returns with a single space, and
    collapse repeated whitespace, so the value is safe to place in a
    tab-separated field.
    """
    if text is None:
        return ""
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main conversion function
# ---------------------------------------------------------------------------

def json_to_tsv(
    input_path,
    output=None,
    domain="cs",
    progress_every=100_000,
):
    """
    Stream an arXiv JSON-lines file and write a tab-separated file with:

        id, title, abstract, keywords, <field1>, ..., <fieldN>

    Parameters
    ----------
    input_path : str
        Path to the input JSON-lines file (e.g. output of
        extract_arxiv_category.py).
    output : str or None
        Path to the output .tsv file. Defaults to
        "<input_path_without_ext>.tsv".
    domain : str
        Which predefined domain constant to use when `fields` is None.
        One of: "cs" (default), "math", "stat", "q-bio", "q-fin", "eess",
        "physics".
    progress_every : int
        Print progress every N records processed.

    Returns
    -------
    (n_read, n_written) : tuple[int, int]
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fields = DOMAIN_FIELDS[domain]

    if output is None:
        base, _ = os.path.splitext(input_path)
        output = base + ".tsv"

    header = ["id", "title", "abstract"] + fields

    n_read = 0
    n_written = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output, "w", encoding="utf-8", newline="") as fout:

        fout.write("\t".join(header) + "\n")

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_read += 1

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            paper_id = clean_text(str(paper.get("id", "")))
            title = clean_text(paper.get("title", ""))
            abstract = clean_text(paper.get("abstract", ""))

            categories_str = paper.get("categories", "") or ""
            categories_list = categories_str.split()

            binary_cols = [
                "1" if cat in categories_list else "0" for cat in fields
            ]

            row = [paper_id, title, abstract] + binary_cols
            fout.write("\t".join(row) + "\n")
            n_written += 1

            if progress_every and n_read % progress_every == 0:
                print(f"...processed {n_read:,} records", file=sys.stderr)

    print(f"Done. Read {n_read:,} records, wrote {n_written:,} rows to "
          f"{output}", file=sys.stderr)

    return n_read, n_written

