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

import re
from typing import List, Dict, Optional

from . import Work, AuthorCandidate
from .tools import fetch_page, web_search

from bs4 import BeautifulSoup

IEEE_AUTHOR = "ieeexplore.ieee.org/author"

def get_ids(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate IEEE Xplore author profiles by name, via web search.
    IEEE's own API has no author-search/lookup-by-id endpoint, so this
    finds the numeric author id (e.g. "37087217367") from author profile
    page URLs (ieeexplore.ieee.org/author/{id}) surfaced in search results.

    affiliation/email_domain/interests are not derivable from search
    snippets and are always None / [] here.
    """
    query = f'"{name}" site:{IEEE_AUTHOR}'
    raw_results = web_search(query)

    candidates: List[AuthorCandidate] = []
    seen_ids = set()

    for res in raw_results:
        url = res.get("url", "")
        match = re.search(r"ieeexplore\.ieee\.org/author/(\d+)", url)
        if not match:
            continue
        author_id = match.group(1)
        if author_id in seen_ids:
            continue
        seen_ids.add(author_id)

        candidates.append(
            AuthorCandidate(id=author_id, name=res.get("title", name))
        )
        if len(candidates) >= max_results:
            break

    return candidates


def get_interests(author_id) -> List[str]:
    """
    Parse "Publication Topics" (interests) from an IEEE Xplore author
    profile page's HTML.

    Args:
        html: full page HTML (e.g. from a browser render/headless fetch,
            since IEEE Xplore blocks direct server-side scraping and the
            topics are client-side rendered)

    Returns:
        List of topic strings, e.g. ["Language Model", "Diffusion Model", ...]
        Empty list if the keywords container isn't found.
    """

    url = f"https://{IEEE_AUTHOR}/{author_id}"

    resp = fetch_page(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "html.parser")

    container = soup.find("div", class_="keywords-container")
    if container is None:
        return []

    interests = []
    for a in container.find_all("a"):
        text = a.get_text(strip=True)
        if text:
            interests.append(text.rstrip(","))

    return interests