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


# ---------------------------------------------------------------------------
# Domain category constants
# ---------------------------------------------------------------------------

# Computer Science -- the 40 official cs.* categories (most complete/reliable)
CS_FIELDS = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
]

# Mathematics
MATH_FIELDS = [
    "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO",
    "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
    "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT",
    "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA",
    "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG",
    "math.SP", "math.ST",
]

# Statistics
STAT_FIELDS = ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"]

# Quantitative Biology
QBIO_FIELDS = [
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
    "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
]

# Quantitative Finance
QFIN_FIELDS = [
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
    "q-fin.RM", "q-fin.ST", "q-fin.TR",
]

# Electrical Engineering and Systems Science
EESS_FIELDS = ["eess.AS", "eess.IV", "eess.SP", "eess.SY"]

# Physics -- physics doesn't follow the same "physics.XX" pattern for
# everything; many sub-areas are their own top-level archives. Listed here
# as best-effort; adjust as needed for your use case.
PHYSICS_FIELDS = [
    "astro-ph", "cond-mat", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math-ph", "nlin", "nucl-ex", "nucl-th", "quant-ph",
    "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus",
    "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
    "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
    "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det",
    "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph",
    "physics.soc-ph", "physics.space-ph",
]

DOMAIN_FIELDS = {
    "cs": CS_FIELDS,
    "math": MATH_FIELDS,
    "stat": STAT_FIELDS,
    "q-bio": QBIO_FIELDS,
    "q-fin": QFIN_FIELDS,
    "eess": EESS_FIELDS,
    "physics": PHYSICS_FIELDS,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"[\n\t\r]+")


def _clean_text(text):
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
    fields=None,
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
    fields : list[str] or None
        List of category codes to create binary (1/0) columns for, e.g.
        ["cs.LG", "cs.CV"]. If None, uses DOMAIN_FIELDS[domain].
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

    if fields is None:
        if domain not in DOMAIN_FIELDS:
            raise ValueError(
                f"Unknown domain '{domain}'. Known domains: "
                f"{list(DOMAIN_FIELDS.keys())}. Or pass `fields` explicitly."
            )
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

            paper_id = _clean_text(str(paper.get("id", "")))
            title = _clean_text(paper.get("title", ""))
            abstract = _clean_text(paper.get("abstract", ""))

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
