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

import requests
import json
from typing import List, Optional

from . import Work, AuthorCandidate

BASE_URL = "https://pub.orcid.org/v3.0/"
HEADERS = {"Accept": "application/json"}

def get_ids(given: str, family: str, max_results: Optional[int] = None) -> List[AuthorCandidate]:
    """
    Search ORCID for a person by given/family name and return candidate
    matches with enough info to disambiguate: orcid id, name, and
    institution affiliations.

    Args:
        given: given (first) name, e.g. "Jason"
        family: family (last) name, e.g. "Priem"
        max_results: cap on number of results returned

    Returns:
        List of AuthorCandidate, ordered by ORCID's relevance ranking.
        affiliation is the first listed institution, when ORCID has any
        on record. email_domain and interests are always None / [] —
        ORCID doesn't expose either via this endpoint.
    """
    query = f'given-names:{given} AND family-name:{family}'
    r = requests.get(
        f"{BASE_URL}expanded-search",
        params={"q": query, "rows": max_results or 20},
        headers={"Accept": "application/json"},
    )
    r.raise_for_status()
    results = r.json().get("expanded-result") or []

    candidates: List[AuthorCandidate] = []
    for res in results:
        institutions = res.get("institution-name") or []
        display_name = res.get("credit-name") or f"{res.get('given-names', '')} {res.get('family-names', '')}".strip()

        candidates.append(
            AuthorCandidate(
                id=res.get("orcid-id", ""),
                name=display_name,
                affiliation=institutions[0] if institutions else None,
            )
        )

    return candidates

def get_works(author_id: str, url: Optional[str] = None, batch_size: int = 50) -> List[Work]:
    """
    Fetch all works for a given ORCID iD.

    ORCID's /works endpoint returns grouped summaries (no cursor paging needed —
    it's a single call), but each summary is shallow. This function follows up
    with the bulk /works/{put-codes} endpoint to get full details, batching
    put-codes to avoid overly long URLs.

    If `url` is given, the raw list of full work records is also saved there as JSON.

    Args:
        author_id: ORCID iD, e.g. "0000-0002-1825-0097"
        url: optional file path to also dump the raw ORCID work JSON to
        batch_size: number of put-codes to fetch per bulk request

    Returns:
        List of Work. ORCID rarely exposes abstracts or keywords, so those
        fields are usually "" / [] respectively.
    """
    # Step 1: get all work summaries (grouped, deduplicated)
    resp = requests.get(f"{BASE_URL}{author_id}/works", headers=HEADERS)
    resp.raise_for_status()
    groups = resp.json().get("group", [])

    # Each group can have multiple summaries (same work from different sources);
    # take the put-code of the first summary in each group.
    put_codes = []
    for group in groups:
        summaries = group.get("work-summary", [])
        if summaries:
            put_codes.append(str(summaries[0]["put-code"]))

    if not put_codes:
        return []

    # Step 2: fetch full details in batches via the bulk endpoint
    raw_works = []
    for i in range(0, len(put_codes), batch_size):
        batch = put_codes[i:i + batch_size]
        codes_param = ",".join(batch)
        r = requests.get(f"{BASE_URL}{author_id}/works/{codes_param}", headers=HEADERS)
        r.raise_for_status()
        bulk = r.json().get("bulk", [])
        for item in bulk:
            work = item.get("work")
            if work:
                raw_works.append(work)

    if url:
        with open(url, "w", encoding="utf-8") as f:
            json.dump(raw_works, f, indent=2, ensure_ascii=False)

    works: List[Work] = []
    for w in raw_works:
        title = (w.get("title") or {}).get("title", {}).get("value")

        abstract = w.get("short-description") or ""

        link = (w.get("url") or {}).get("value")

        contributors = (w.get("contributors") or {}).get("contributor", [])
        authors = [
            (c.get("credit-name") or {}).get("value")
            for c in contributors
        ]
        authors = [a for a in authors if a]

        venue = (w.get("journal-title") or {}).get("value")

        pub_date = w.get("publication-date") or {}
        year_field = pub_date.get("year") or {}
        year = year_field.get("value")

        if venue and year:
            venue = f"{venue} {year}"
        elif not venue and year:
            venue = str(year)

        # Prefer a DOI as the id when available (more portable/stable than
        # put-code, which is only meaningful within ORCID); fall back to
        # put-code otherwise.
        put_code = str(w.get("put-code", ""))
        work_id = put_code
        for ext_id in (w.get("external-ids") or {}).get("external-id", []):
            if ext_id.get("external-id-type") == "doi":
                work_id = ext_id.get("external-id-value", put_code)
                break

        works.append(
            Work(
                id=work_id,
                title=title,
                year=year,
                abstract=abstract,
                link=link,
                authors=authors,
                keywords=[],  # ORCID does not provide keywords per-work
                language=w.get("language-code"),
                venue=venue,
            )
        )

    return works