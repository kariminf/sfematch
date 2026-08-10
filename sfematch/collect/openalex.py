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

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Union, List

import requests

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


def get_ids(name: str, per_page: int = 5) -> List[str]:
    """
    Search OpenAlex for an author by name and return a list of OpenAlex IDs
    for all matches (e.g. ['https://openalex.org/A5023888391', ...]).

    Returns an empty list if nothing was found.
    """
    resp = requests.get(
        f"{BASE_URL}/authors",
        params={**_common_params(), "search": name, "per-page": per_page},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    return [r["id"] for r in results]

def get_candidates(name: str, per_page: int = 5) -> List[dict]:
    """
    Returns a list of candidate matches with enough info to disambiguate:
    id, display_name, and last known institution.
    """
    resp = requests.get(
        f"{BASE_URL}/authors",
        params={**_common_params(), "search": name, "per-page": per_page},
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    return [
        {
            "id": r["id"].split("/")[-1],
            "display_name": r.get("display_name"),
            "institution": (r.get("last_known_institution") or {}).get("display_name"),
            "works_count": r.get("works_count"),
        }
        for r in results
    ]

def get_works(author_id: str, url: Optional[str] = None, per_page: int = 100) -> List[dict]:
    """
    Fetch all works for a given OpenAlex author id (paginates automatically
    via cursor paging, so you get everything, not just the first page).

    If `url` is given, the raw list of works is also saved there as JSON.

    Returns a list of raw OpenAlex "work" dicts.
    """
    author_filter = _short_id(author_id)

    works: list[dict] = []
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

        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")

    if url:
        with open(url, "w", encoding="utf-8") as f:
            json.dump(works, f, ensure_ascii=False, indent=2)

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


@dataclass
class Paper:
    id: str
    title: Optional[str]
    abstract: str
    link: Optional[str]
    authors: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    language: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


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


def extract_works(works_data: Union[list[dict], dict]) -> list[Paper]:
    """
    Take a works collection (either the list returned by get_works, or a raw
    OpenAlex response dict with a "results" key) and extract a minimal,
    flat representation of each paper.

    Returns a list of Paper objects with: id, title, abstract, link,
    authors, keywords, language.
    """
    if isinstance(works_data, dict):
        works_data = works_data.get("results", [])

    works: list[Paper] = []

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
            Paper(
                id=w.get("id"),
                title=w.get("title"),
                abstract=construct_abstract(w.get("abstract_inverted_index")),
                link=_primary_link(w),
                authors=authors,
                keywords=keywords,
                language=w.get("language"),
            )
        )

    return works
