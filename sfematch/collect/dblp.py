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
import xml.etree.ElementTree as ET

from typing import List, Optional

from . import Work, AuthorCandidate


BASE_URL = "https://dblp.org/pid/"


def get_id(name: str) -> Optional[str]:
    """
    Look up a DBLP pid for a given full name.
    Returns the top-matching pid, or None if no match is found.

    Warning: names are often ambiguous (multiple authors share a name).
    Use get_ids() instead if you want to see/disambiguate all candidates.
    """
    candidates = get_ids(name)
    return candidates[0]["id"] if candidates else None

    
def get_ids(name: str, max_results: Optional[int] = None) -> List[AuthorCandidate]:
    """
    Search DBLP for authors matching a name.

    Args:
        name: full name to search, e.g. "Jiawei Han"
        max_results: cap on number of results (DBLP's `h` param controls
            how many hits are fetched; None uses DBLP's default of 20)

    Returns:
        List of AuthorCandidate, ordered by DBLP's relevance ranking.
        DBLP doesn't expose affiliation, email_domain, or interests via
        this endpoint, so those are always None / [].
    """
    url = "https://dblp.org/search/author/api"
    params = {"q": name, "format": "json", "h": max_results or 20}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    results: List[AuthorCandidate] = []
    for hit in hits:
        info = hit.get("info", {})
        profile_url = info.get("url", "")
        # profile_url looks like: https://dblp.org/pid/97/4260.html
        pid = None
        if "/pid/" in profile_url:
            pid = profile_url.split("/pid/")[1]
            if pid.endswith(".html"):
                pid = pid[: -len(".html")]
            pid = pid.rstrip("/")

        results.append(
            AuthorCandidate(
                id= pid or "",
                name=info.get("author", ""),
                affiliation=None,
                email=None,
                interests=[],
            )
        )

    return results

BASE_URL = "https://dblp.org/pid/"


def get_works(dblp_id: str) -> List[Work]:
    """
    Fetch and parse publications for a DBLP person ID (pid).

    Args:
        dblp_id: DBLP pid, e.g. "97/4260" (as it appears in
                 https://dblp.org/pid/97/4260.xml or the person's profile URL)

    Returns:
        List of Work. DBLP doesn't expose abstracts, keywords, or language,
        so those fields are always "" / [] / None respectively.
    """
    dblp_id = dblp_id.strip().strip("/")
    url = f"{BASE_URL}{dblp_id}.xml"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    works: List[Work] = []
    # Each <r> wraps one publication entry (article, inproceedings, etc.)
    for r in root.findall("r"):
        pub = list(r)[0]  # the actual element: article/inproceedings/incollection/...

        title_el = pub.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else None

        authors = [a.text for a in pub.findall("author") if a.text]

        # journal for articles, booktitle for conference papers
        venue_el = pub.find("journal")
        if venue_el is None:
            venue_el = pub.find("booktitle")
        venue = venue_el.text if venue_el is not None else None

        year_el = pub.find("year")
        year = year_el.text if year_el is not None else None
        if venue and year:
            venue = f"{venue} {year}"

        # Prefer the official DBLP record URL; fall back to first <ee> (electronic edition)
        url_el = pub.find("url")
        ee_el = pub.find("ee")
        if url_el is not None and url_el.text:
            link = "https://dblp.org/" + url_el.text
        elif ee_el is not None and ee_el.text:
            link = ee_el.text
        else:
            link = None

        # DBLP keys (e.g. "conf/nips/VaswaniSPUJGKP17") are the natural
        # unique id for a publication, more stable than title/venue combos
        key = pub.attrib.get("key", "")

        works.append(
            Work(
                id=key,
                title=title,
                year=year,
                abstract="",  # DBLP does not provide abstracts
                link=link,
                authors=authors,
                keywords=[],  # DBLP does not provide keywords
                language=None,  # DBLP does not provide language
                venue=venue,
            )
        )

    return works





if __name__ == "__main__":
    # Example: Alan Turing-esque test — replace with a real pid
    results = get_works("97/4260")
    for w in results[:5]:
        print(w)

# def get_id(name: str) -> Optional[str]:
