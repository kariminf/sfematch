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

# No API

import re

from typing import List

from . import AuthorCandidate
from .tools import web_search

RESEARCHGATE_PROFILE = "researchgate.net/profile"


def get_ids(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate ResearchGate author profiles by name, via web search.

    ResearchGate has no public API and disallows direct scraping (ToS +
    active bot-blocking), so this finds candidates the same way as the
    IEEE/ScienceDirect/Scholar modules: search-engine results pointing at
    profile pages, then extract the stable slug from the URL.

    Profile URLs look like:
        https://www.researchgate.net/profile/Yann-Lecun
        https://www.researchgate.net/profile/Yann-Lecun-2   (disambiguated)
    The path segment after "/profile/" is the id.

    affiliation is best-effort, parsed from the search snippet/title when
    present (ResearchGate titles/snippets often look like
    "Yann LECUN | New York University, New York | NYU | ..."). 
    email_domain and interests are always None / [] — not derivable from
    search results.

    Args:
        name: full name to search, e.g. "Yann LeCun"
        max_results: cap on number of candidates returned

    Returns:
        List of AuthorCandidate. Because this relies on search snippets
        rather than a real API, results are noisier and less reliable than
        arXiv/DBLP/OpenAlex/ORCID — treat as a lead to verify, not ground
        truth.
    """
    query = f'"{name}" site:{RESEARCHGATE_PROFILE}'
    raw_results = web_search(query)

    candidates: List[AuthorCandidate] = []
    seen_ids = set()

    for res in raw_results:
        url = res.get("url", "")
        match = re.search(r"researchgate\.net/profile/([^/?#]+)", url)
        if not match:
            continue
        profile_id = match.group(1)
        if profile_id in seen_ids:
            continue
        seen_ids.add(profile_id)

        title = res.get("title", "")
        # Titles look like: "Yann LECUN | New York University, New York | NYU"
        parts = [p.strip() for p in title.split("|")]
        display_name = parts[0] if parts and parts[0] else name
        affiliation = parts[1] if len(parts) >= 2 and parts[1] else None

        candidates.append(
            AuthorCandidate(id=profile_id, name=display_name, affiliation=affiliation)
        )
        if len(candidates) >= max_results:
            break

    return candidates
