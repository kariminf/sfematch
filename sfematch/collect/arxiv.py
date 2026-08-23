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
import sys

import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from . import Work

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

def get_works(name: str, max_results: int = 50) -> List[Work]:
    """
    Fetch works from arXiv authored by `name` (full name), using arXiv's
    Atom API search interface.
    """
    base_url = "http://export.arxiv.org/api/query"
    query = f'au:"{name}"'
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        raw = response.read()

    root = ET.fromstring(raw)
    works: List[Work] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = _text(entry, f"{ATOM_NS}id")
        # arXiv id URLs look like http://arxiv.org/abs/XXXX.XXXXXvN
        short_id = arxiv_id.rsplit("/", 1)[-1] if arxiv_id else ""

        title = _text(entry, f"{ATOM_NS}title")
        title = " ".join(title.split()) if title else None

        abstract = _text(entry, f"{ATOM_NS}summary")
        abstract = " ".join(abstract.split()) if abstract else ""

        link = None
        for link_el in entry.findall(f"{ATOM_NS}link"):
            if link_el.attrib.get("type") == "text/html":
                link = link_el.attrib.get("href")
                break
        if link is None:
            link = arxiv_id

        authors = [
            _text(author, f"{ATOM_NS}name")
            for author in entry.findall(f"{ATOM_NS}author")
        ]
        authors = [a for a in authors if a]

        keywords = [
            cat.attrib.get("term")
            for cat in entry.findall(f"{ATOM_NS}category")
            if cat.attrib.get("term")
        ]

        venue = _text(entry, f"{ARXIV_NS}journal_ref")
        venue = " ".join(venue.split()) if venue else "arXiv"

        year = _text(entry, f"{ATOM_NS}published")[:4] if _text(entry, f"{ATOM_NS}published") else None

        works.append(
            Work(
                id=short_id,
                title=title,
                year=None,
                abstract=abstract,
                link=link,
                authors=authors,
                keywords=keywords,
                language=None,  # arXiv metadata does not provide language
                venue=venue,
            )
        )

    return works


def _text(elem: ET.Element, tag: str) -> Optional[str]:
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else None



def _matches_prefix(categories_str, prefix, primary_only=False):
    """
    Return True if the paper's categories string contains at least one
    category code starting with `prefix` (or, if primary_only=True,
    if its FIRST/primary category starts with `prefix`).
    """
    codes = categories_str.split()
    if not codes:
        return False
    if primary_only:
        return codes[0].startswith(prefix)
    return any(code.startswith(prefix) for code in codes)


def extract_category(
    prefix,
    source="./arxiv-metadata-oai-snapshot.json",
    output=None,
    primary_only=False,
    fields=None,
    progress_every=100_000,
):
    """
    Stream through the arXiv metadata snapshot and write out only the
    papers whose category codes match `prefix`.

    Parameters
    ----------
    prefix : str
        Category prefix to match, e.g. "cs.", "cs.LG", "physics", "math.".
    source : str
        Path to the input JSON-lines snapshot file. Defaults to
        "./arxiv-metadata-oai-snapshot.json".
    output : str or None
        Path to the output JSON-lines file. If None, defaults to
        "<prefix-sanitized>_papers.jsonl" in the current directory.
    primary_only : bool
        If True, only match papers whose PRIMARY (first-listed) category
        starts with `prefix`. If False (default), match papers that have
        ANY category starting with `prefix`.
    fields : list[str] or None
        If given, only keep these fields per record (e.g.
        ["title", "abstract", "authors", "categories"]) to shrink output
        size. If None, keep the full original record.
    progress_every : int
        Print a progress line every N input records read.

    Returns
    -------
    (n_read, n_written) : tuple[int, int]
    """
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Source file not found: {source}")

    if output is None:
        safe_prefix = prefix.strip().rstrip(".").replace(".", "_") or "all"
        output = f"{safe_prefix}_papers.jsonl"

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

            if _matches_prefix(categories_str, prefix, primary_only):
                if fields is not None:
                    record = {k: paper.get(k) for k in fields}
                else:
                    record = paper

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

            if progress_every and n_read % progress_every == 0:
                print(f"...read {n_read:,} records, matched {n_written:,}",
                      file=sys.stderr)

    print(f"Done. Read {n_read:,} records from {source}, "
          f"wrote {n_written:,} matching records to {output}",
          file=sys.stderr)

    return n_read, n_written

