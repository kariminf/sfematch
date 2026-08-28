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
collect_openalex.py

Small helper module to search authors on OpenAlex, pull their works,
and extract a clean, minimal representation of each paper.

OpenAlex API docs: https://docs.openalex.org/
No API key is required. It's polite (and gets you into a faster pool)
to identify yourself via a `mailto` param — set OPENALEX_EMAIL below
if you want that.
"""

import json
import requests

from pathlib import Path
from typing import Optional, Union, List

from . import Work, AuthorCandidate

BASE_URL = "https://api.openalex.org"

# Optional: set this to your email to use OpenAlex's "polite pool"
# (faster, more reliable rate limits). Leave as None to skip it.
OPENALEX_EMAIL: Optional[str] = None


def _common_params() -> dict:
    """Params to attach to every request (e.g. polite-pool email)."""
    params = {}
    if OPENALEX_EMAIL:
        params["mailto"] = OPENALEX_EMAIL
    return params


def _short_id(openalex_id: str) -> str:
    """Turn 'https://openalex.org/A123' into 'A123' (also accepts short ids as-is)."""
    if not openalex_id:
        return openalex_id
    return openalex_id.rstrip("/").split("/")[-1]


def get_ids(name: str, per_page: int = 5) -> List[AuthorCandidate]:
    """
    Search OpenAlex for authors matching a name.

    Args:
        name: full name to search, e.g. "Jason Priem"
        per_page: cap on number of results returned

    Returns:
        List of AuthorCandidate, ordered by OpenAlex's relevance ranking.
        affiliation is the author's last known institution, when OpenAlex
        has one on record. email_domain and interests are always None / []
        — OpenAlex doesn't expose either.
    """
    resp = requests.get(
        f"{BASE_URL}/authors",
        params={**_common_params(), "search": name, "per-page": per_page},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    return [
        AuthorCandidate(
            id=r["id"].split("/")[-1],
            name=r.get("display_name", ""),
            affiliation=(r.get("last_known_institution") or {}).get("display_name"),
        )
        for r in results
    ]

def get_works(author_id: str, url: Optional[str] = None, per_page: int = 100) -> List[Work]:

    author_filter = _short_id(author_id)

    raw_works: list[dict] = []
    cursor = "*"

    while cursor:
        resp = requests.get(
            f"{BASE_URL}/works",
            params={
                **_common_params(),
                "filter": f"author.id:{author_filter}",
                "per-page": per_page,
                "cursor": cursor,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        raw_works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")

    if url:
        with open(url, "w", encoding="utf-8") as f:
            json.dump(raw_works, f, ensure_ascii=False, indent=2)

    works = extract_works(raw_works)

    return works


def construct_abstract(inverted_index: Optional[dict]) -> str:
    """
    OpenAlex stores abstracts as an inverted index: {word: [positions...]}.
    This rebuilds the plain-text abstract from that structure.
    """
    if not inverted_index:
        return ""

    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word

    return " ".join(positions[i] for i in sorted(positions))



def _primary_link(work: dict) -> Optional[str]:
    """Pick a single, best link for the work (landing page > OA url > OpenAlex id)."""
    primary_location = work.get("primary_location") or {}
    link = primary_location.get("landing_page_url")
    if link:
        return link

    open_access = work.get("open_access") or {}
    if open_access.get("oa_url"):
        return open_access["oa_url"]

    return work.get("id")


def extract_works(works_data: Union[list[dict], dict]) -> list[Work]:
    """
    Take a works collection (either the list returned by get_works, or a raw
    OpenAlex response dict with a "results" key) and extract a minimal,
    flat representation of each paper.

    Returns a list of Paper objects with: id, title, abstract, link,
    authors, keywords, language.
    """
    if isinstance(works_data, dict):
        works_data = works_data.get("results", [])

    works: list[Work] = []

    for w in works_data:
        authors = [
            a["author"]["display_name"]
            for a in w.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ]

        # Prefer OpenAlex's dedicated "keywords" field; fall back to concepts.
        if w.get("keywords"):
            keywords = [k.get("display_name") for k in w["keywords"] if k.get("display_name")]
        else:
            keywords = [c.get("display_name") for c in w.get("concepts", []) if c.get("display_name")]

        

        works.append(
            Work(
                id=_short_id(w.get("id")),
                title=w.get("title"),
                year = w.get("publication_year"),
                abstract=construct_abstract(w.get("abstract_inverted_index")),
                link=_primary_link(w),
                authors=authors,
                keywords=keywords,
                language=w.get("language"),
            )
        )

    return works


def format_index(index: dict[str, list[dict]]) -> str:
    """
    Pretty-print the index, but keep each {"id":..,"pos":..,"nbr":..}
    entry on a single line instead of one line per key.
    """
    lines = ["{"]
    items = list(index.items())

    for i, (name, works) in enumerate(items):
        lines.append(f"  {json.dumps(name, ensure_ascii=False)}: [")
        for j, w in enumerate(works):
            comma = "," if j < len(works) - 1 else ""
            lines.append(f"    {json.dumps(w, ensure_ascii=False)}{comma}")
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"  ]{comma}")

    lines.append("}")
    return "\n".join(lines)


def index_authors(works_path, out_file):
    index: dict[str, list[dict]] = {}

    for pub_file in Path(works_path).glob("*.json"):
        with open(pub_file, encoding="utf-8") as f:
            work = json.load(f)

        authors = work.get("authors", [])
        if not authors:
            continue

        work_id = _short_id(work.get("id", pub_file.stem))
        nbr = len(authors)

        for pos, name in enumerate(authors, start=1):
            index.setdefault(name, []).append({
                "id": work_id,
                "pos": pos,
                "nbr": nbr,
            })

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(format_index(index))

    print(f"Indexed {len(index)} authors across {sum(len(v) for v in index.values())} authorships")
    print(f"Saved to {out_file}")