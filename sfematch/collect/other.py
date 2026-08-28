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
from .tools import fetch_page, web_search


def get_github_ids(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate GitHub profiles by name, via web search.

    GitHub does have a real public REST API for user search
    (api.github.com/search/users), which would return far more reliable,
    structured results (bio, company, blog url, etc.) than scraping search
    snippets — this web-search version is kept for consistency with the
    other no-API modules (ResearchGate, IEEE, ScienceDirect, Scholar), but
    consider switching to the GitHub API if this becomes a bottleneck.
    """
    query = f'"{name}" site:github.com/'
    raw_results = web_search(query)

    candidates: List[AuthorCandidate] = []
    seen_ids = set()

    for res in raw_results:
        url = res.get("url", "")
        match = re.search(r"github.com/([^/?#]+)", url)
        if not match:
            continue
        profile_id = match.group(1)
        if profile_id in seen_ids:
            continue
        seen_ids.add(profile_id)

        title = res.get("title", "")
        parts = [p.strip() for p in title.split("|")]
        display_name = parts[0] if parts and parts[0] else name
        affiliation = parts[1] if len(parts) >= 2 and parts[1] else None

        candidates.append(
            AuthorCandidate(id=profile_id, name=display_name, affiliation=affiliation)
        )
        if len(candidates) >= max_results:
            break

    return candidates

def get_linkedin_ids(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate LinkedIn profiles by name, via web search.

    LinkedIn has no public search API for this and blocks direct scraping
    aggressively, so this is search-snippet-based like the other no-API
    modules — treat results as leads to verify, not ground truth.
    """
    query = f'"{name}" site:linkedin.com/in/'
    raw_results = web_search(query)

    candidates: List[AuthorCandidate] = []
    seen_ids = set()

    for res in raw_results:
        url = res.get("url", "")
        match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
        if not match:
            continue
        profile_id = match.group(1)
        if profile_id in seen_ids:
            continue
        seen_ids.add(profile_id)

        title = res.get("title", "")
        parts = [p.strip() for p in title.split("|")]
        display_name = parts[0] if parts and parts[0] else name
        affiliation = parts[1] if len(parts) >= 2 and parts[1] else None

        candidates.append(
            AuthorCandidate(id=profile_id, name=display_name, affiliation=affiliation)
        )
        if len(candidates) >= max_results:
            break

    return candidates

def get_websites(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate personal/homepage websites by name, via web search.

    Unlike the profile-based modules (GitHub, LinkedIn, ResearchGate, ...),
    there's no fixed URL pattern to match against, so this is a much
    broader/noisier search. `id` is the full result URL (not just the bare
    domain) — using only the domain would incorrectly merge distinct pages
    that happen to share a host, e.g. two different people's pages under
    the same university domain or the same blogging platform
    (medium.com/@alice vs medium.com/@bob).
    """
    query = f'"{name}" website'
    raw_results = web_search(query)

    candidates: List[AuthorCandidate] = []
    seen_ids = set()

    for res in raw_results:
        url = res.get("url", "")
        if not url:
            continue
        if url in seen_ids:
            continue
        seen_ids.add(url)

        title = res.get("title", "")
        parts = [p.strip() for p in title.split("|")]
        display_name = parts[0] if parts and parts[0] else name
        affiliation = parts[1] if len(parts) >= 2 and parts[1] else None

        candidates.append(
            AuthorCandidate(id=url, name=display_name, affiliation=affiliation)
        )
        if len(candidates) >= max_results:
            break

    return candidates