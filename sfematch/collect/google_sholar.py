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
from dataclasses import dataclass, field, asdict
from typing import Optional, List

from scholarly import scholarly, ProxyGenerator


from . import Work, AuthorCandidate
from .tools import web_search

# pg = ProxyGenerator()
# success = pg.FreeProxies()  # free, but slow/unreliable
# scholarly.use_proxy(pg)

SCHOLAR_AUTHOR = "scholar.google.com/citations"
 
 
def get_ids(name: str, max_results: int = 5) -> List[AuthorCandidate]:
    """
    Find candidate Google Scholar author profiles by name, via web search.
 
    Google Scholar has no public author-search API (unofficial scraping
    libraries like `scholarly` exist but are fragile and frequently
    blocked), so this finds candidates the same way as the IEEE/
    ScienceDirect modules: search-engine results pointing at profile
    pages, then extract the stable id from the URL.
 
    Scholar profile URLs look like:
        https://scholar.google.com/citations?user=WLN3QrAAAAAJ&hl=en
    The "user" query-param value is the author id.
 
    affiliation/email_domain/interests are not derivable from search
    snippets and are always None / [] here.
    """
    query = f'"{name}" site:{SCHOLAR_AUTHOR}'
    raw_results = web_search(query)
 
    candidates: List[AuthorCandidate] = []
    seen_ids = set()
 
    for res in raw_results:
        url = res.get("url", "")
        match = re.search(r"[?&]user=([\w-]+)", url)
        if not match:
            continue
        author_id = match.group(1)
        if author_id in seen_ids:
            continue
        seen_ids.add(author_id)
 
        title = res.get("title", "")
        # Scholar result titles look like: "Yann LeCun - ‪Google Scholar‬"
        display_name = title.split(" - ")[0].strip() if " - " in title else title or name
 
        # Snippets often look like: "Professor of Computer Science, NYU - ..."
        # so treat the snippet's leading clause (before a comma) as a
        # best-effort affiliation guess.
        snippet = res.get("snippet", "")
        affiliation = snippet.split(" - ")[0].strip() if snippet else None
 
        candidates.append(
            AuthorCandidate(id=author_id, name=display_name, affiliation=affiliation)
        )
        if len(candidates) >= max_results:
            break
 
    return candidates


# def get_ids(name: str, max_results: Optional[int] = None) -> List[AuthorCandidate]:
#     """
#     Look up Google Scholar author profiles matching `name` and return
#     candidate info for each (scholar_id plus disambiguating metadata).

#     Returns an empty list if no matches are found.
#     """
#     search_query = scholarly.search_author(name)

#     candidates: List[AuthorCandidate] = []
#     for i, author in enumerate(search_query):
#         if max_results is not None and i >= max_results:
#             break
#         candidates.append(
#             AuthorCandidate(
#                 id=author.get("scholar_id", ""),
#                 name=author.get("name", ""),
#                 affiliation=author.get("affiliation"),
#                 email=author.get("email_domain"),
#                 interests=author.get("interests", []) or [],
#             )
#         )

#     return candidates


def get_works(author_id: str, max_results: Optional[int] = None) -> List[Work]:
    """
    Fetch publications for a Google Scholar author, given their scholar_id
    (as returned by get_id).
    """
    author = scholarly.search_author_id(author_id)
    author = scholarly.fill(author, sections=["publications"])

    publications = author.get("publications", [])
    if max_results is not None:
        publications = publications[:max_results]

    works: List[Work] = []
    for pub in publications:
        # Publication summaries from search_author are shallow; fill() each
        # one to get title, abstract, authors, etc.
        try:
            pub = scholarly.fill(pub)
        except Exception:
            # Some entries fail to fill (e.g. rate limiting, missing pages);
            # skip rather than crash the whole batch.
            continue

        bib = pub.get("bib", {})

        pub_id = pub.get("author_pub_id") or bib.get("cites_id") or ""
        title = bib.get("title")
        abstract = bib.get("abstract", "") or ""
        link = pub.get("pub_url") or pub.get("eprint_url")

        authors_field = bib.get("author", "")
        authors = [a.strip() for a in authors_field.split(" and ") if a.strip()]

        year = bib.get("year", "") or None

        works.append(
            Work(
                id=pub_id,
                title=title,
                year=year,
                abstract=abstract,
                link=link,
                authors=authors,
                keywords=[],  # Scholar doesn't expose keywords/tags
                language=None,  # Scholar doesn't expose language
            )
        )

    return works


if __name__ == "__main__":
    scholar_id = get_id("Yoshua Bengio")
    if scholar_id:
        results = get_works(scholar_id, max_results=5)
        for w in results:
            print(w.id, "-", w.title)
    else:
        print("Author not found")