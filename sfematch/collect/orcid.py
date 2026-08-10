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

BASE_URL = "https://pub.orcid.org/v3.0/"
HEADERS = {"Accept": "application/json"}

def get_ids(given: str, family: str) -> List[str]:
    query = f'given-names:{given} AND family-name:{family}'
    r = requests.get(f"{BASE_URL}search", params={"q": query}, headers={"Accept": "application/json"})
    results = r.json().get("result") or []
    return [res["orcid-identifier"]["path"] for res in results]

def get_candidates(given: str, family: str, rows: int = 5) -> List[dict]:
    """
    Search ORCID for a person by given/family name and return candidate
    matches with enough info to disambiguate: orcid id, name, and
    institution affiliations.
    """
    query = f'given-names:{given} AND family-name:{family}'
    r = requests.get(
        f"{BASE_URL}expanded-search",
        params={"q": query, "rows": rows},
        headers={"Accept": "application/json"},
    )
    r.raise_for_status()
    results = r.json().get("expanded-result") or []

    candidates = []
    for res in results:
        candidates.append({
            "orcid": res.get("orcid-id"),
            "given_names": res.get("given-names"),
            "family_name": res.get("family-names"),
            "credit_name": res.get("credit-name"),
            "institutions": res.get("institution-name") or [],
        })
    return candidates


def get_works(author_id: str, url: Optional[str] = None, batch_size: int = 50) -> List[dict]:
    """
    Fetch all works for a given ORCID iD.

    ORCID's /works endpoint returns grouped summaries (no cursor paging needed —
    it's a single call), but each summary is shallow. This function follows up
    with the bulk /works/{put-codes} endpoint to get full details, batching
    put-codes to avoid overly long URLs.

    If `url` is given, the raw list of full work records is also saved there as JSON.

    Returns a list of raw ORCID "work" dicts.
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
    all_works = []
    for i in range(0, len(put_codes), batch_size):
        batch = put_codes[i:i + batch_size]
        codes_param = ",".join(batch)
        r = requests.get(f"{BASE_URL}{author_id}/works/{codes_param}", headers=HEADERS)
        r.raise_for_status()
        bulk = r.json().get("bulk", [])
        for item in bulk:
            work = item.get("work")
            if work:
                all_works.append(work)

    if url:
        with open(url, "w", encoding="utf-8") as f:
            json.dump(all_works, f, indent=2, ensure_ascii=False)

    return all_works