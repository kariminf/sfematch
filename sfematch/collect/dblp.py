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
from typing import List, Dict, Optional


BASE_URL = "https://dblp.org/pid/"

def get_dblp_works(dblp_id: str) -> List[Dict[str, Optional[str]]]:
    """
    Fetch and parse publications for a DBLP person ID (pid).

    Args:
        dblp_id: DBLP pid, e.g. "97/4260" (as it appears in
                 https://dblp.org/pid/97/4260.xml or the person's profile URL)

    Returns:
        List of dicts with keys: title, authors, year, venue, link
    """
    dblp_id = dblp_id.strip().strip("/")
    url = f"{BASE_URL}{dblp_id}.xml"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    works = []
    # Each <r> wraps one publication entry (article, inproceedings, etc.)
    for r in root.findall("r"):
        pub = list(r)[0]  # the actual element: article/inproceedings/incollection/...

        title_el = pub.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else None

        authors = [a.text for a in pub.findall("author") if a.text]

        year_el = pub.find("year")
        year = year_el.text if year_el is not None else None

        # journal for articles, booktitle for conference papers
        venue_el = pub.find("journal")
        if venue_el is None:
            venue_el = pub.find("booktitle")
        venue = venue_el.text if venue_el is not None else None

        # Prefer the official DBLP record URL; fall back to first <ee> (electronic edition)
        url_el = pub.find("url")
        ee_el = pub.find("ee")
        if url_el is not None and url_el.text:
            link = "https://dblp.org/" + url_el.text
        elif ee_el is not None and ee_el.text:
            link = ee_el.text
        else:
            link = None

        works.append({
            "title": title,
            "authors": authors,
            "year": year,
            "venue": venue,
            "link": link,
        })

    return works


def get_id(name: str) -> Optional[str]:
    """
    Look up a DBLP pid for a given full name.
    Returns the top-matching pid, or None if no match is found.

    Warning: names are often ambiguous (multiple authors share a name).
    Use get_ids() instead if you want to see/disambiguate all candidates.
    """
    candidates = get_ids(name)
    return candidates[0]["pid"] if candidates else None


def get_ids(name: str) -> List[Dict[str, str]]:
    """
    Search DBLP for authors matching a name.

    Args:
        name: full name to search, e.g. "Jiawei Han"

    Returns:
        List of dicts: [{"name": ..., "pid": ..., "url": ...}, ...]
        ordered by DBLP's relevance ranking.
    """
    url = "https://dblp.org/search/author/api"
    params = {"q": name, "format": "json", "h": 20}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits = data.get("result", {}).get("hits", {}).get("hit", [])
    results = []
    for hit in hits:
        info = hit.get("info", {})
        profile_url = info.get("url", "")
        # profile_url looks like: https://dblp.org/pid/97/4260.html
        pid = None
        if "/pid/" in profile_url:
            pid = profile_url.split("/pid/")[1].rstrip(".html").rstrip("/")

        results.append({
            "name": info.get("author"),
            "pid": pid,
            "url": profile_url,
        })

    return results


if __name__ == "__main__":
    # Example: Alan Turing-esque test — replace with a real pid
    results = get_dblp_works("97/4260")
    for w in results[:5]:
        print(w)

# def get_id(name: str) -> Optional[str]:
