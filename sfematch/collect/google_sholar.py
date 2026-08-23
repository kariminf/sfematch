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

from dataclasses import dataclass, field, asdict
from typing import Optional, List

from scholarly import scholarly, ProxyGenerator

from . import Work, AuthorCandidate

pg = ProxyGenerator()
success = pg.FreeProxies()  # free, but slow/unreliable
scholarly.use_proxy(pg)

def get_ids(name: str, max_results: Optional[int] = None) -> List[AuthorCandidate]:
    """
    Look up Google Scholar author profiles matching `name` and return
    candidate info for each (scholar_id plus disambiguating metadata).

    Returns an empty list if no matches are found.
    """
    search_query = scholarly.search_author(name)

    candidates: List[AuthorCandidate] = []
    for i, author in enumerate(search_query):
        if max_results is not None and i >= max_results:
            break
        candidates.append(
            AuthorCandidate(
                id=author.get("scholar_id", ""),
                name=author.get("name", ""),
                affiliation=author.get("affiliation"),
                email=author.get("email_domain"),
                interests=author.get("interests", []) or [],
            )
        )

    return candidates


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