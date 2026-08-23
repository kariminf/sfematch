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
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.prepare.arxiv_fields import *

def _matches_prefix(categories_str, fields=[]):
    """
    Return True if the paper's categories string contains at least one
    category code starting with `prefix` (or, if primary_only=True,
    if its FIRST/primary category starts with `prefix`).
    """
    codes = categories_str.split()
    if not codes:
        return False

    return any((code in fields) for code in codes)


def extract_category(
    prefix,
    fields,
    source="./arxiv-metadata-oai-snapshot.json",
    output=None,
    progress_every=100_000,
):
    """
    Stream through the arXiv metadata snapshot and write out only the
    papers whose category codes match `prefix`.

    Parameters
    ----------
    prefix : str
        Category prefix to match, e.g. "cs.", "cs.LG", "physics", "math.".
    fields : list[str] or None
            If given, only keep these fields per record (e.g.
            ["title", "abstract", "authors", "categories"]) to shrink output
            size. If None, keep the full original record.
    source : str
        Path to the input JSON-lines snapshot file. Defaults to
        "./arxiv-metadata-oai-snapshot.json".
    output : str or None
        Path to the output JSON-lines file. If None, defaults to
        "<prefix-sanitized>_papers.jsonl" in the current directory.
    progress_every : int
        Print a progress line every N input records read.

    Returns
    -------
    (n_read, n_written) : tuple[int, int]
    """
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    if output is None:
        output = f"{prefix}_papers.jsonl"

    n_read = 0
    n_written = 0

    with open(source, "r", encoding="utf-8") as fin, \
         open(output, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_read += 1

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines rather than crashing on a huge file
                continue

            categories_str = paper.get("categories", "") or ""

            if _matches_prefix(categories_str, fields=fields):

                fout.write(json.dumps(paper, ensure_ascii=False) + "\n")
                n_written += 1

            if progress_every and n_read % progress_every == 0:
                print(f"...read {n_read:,} records, matched {n_written:,}",
                      file=sys.stderr)

    print(f"Done. Read {n_read:,} records from {source}, "
          f"wrote {n_written:,} matching records to {output}",
          file=sys.stderr)

    return n_read, n_written


def main():
    parser = argparse.ArgumentParser(
        description="Stream-extract papers matching a category prefix "
                    "(e.g. 'cs.', 'physics', 'math.') from the Kaggle "
                    "arXiv metadata snapshot, without loading the whole "
                    "file into memory."
    )
    parser.add_argument(
        "--dom", default="cs",
        choices=["cs", "math", "stat", "bio", "fin", "eng", "phys"],
        help="Category"
    )
    parser.add_argument(
        "--src", default="./arxiv-metadata-oai-snapshot.json",
        required=True,
        help="Path to the input JSON-lines snapshot file. "
            "Default: './arxiv-metadata-oai-snapshot.json'"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path to the output JSON-lines file. "
            "Default: '<prefix>_papers.jsonl' in the current directory."
    )

    args = parser.parse_args()

    extract_category(
        prefix=args.dom,
        fields=DOMAIN_FIELDS[args.dom],
        source=args.src,
        output=args.out,
    )


if __name__ == "__main__":
    main()
