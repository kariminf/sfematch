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

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.collect.openalex import get_works, index_authors


def process_works(info_url, out_url):

    profiles_path = os.path.join(out_url, "profiles")
    os.makedirs(profiles_path, exist_ok=True)
    works_path = os.path.join(out_url, "works")
    os.makedirs(works_path, exist_ok=True)
    
    with open(info_url, encoding="utf8") as f:
        experts = json.load(f)

    author_works = {}
    for eid in experts:
        author_works[eid] = []

        expert = experts[eid]
        author_id = expert["profiles"]["openalex"]
        if not author_id:
            continue
        out_profile = os.path.join(profiles_path, f"{author_id}.json")
        if os.path.exists(out_profile):
            print(f"{eid} already exists")
            continue

        print(f"Processing {eid} ...")
        works = get_works(author_id, url=out_profile)
        time.sleep(0.5)
        for work in works:
            out_work = os.path.join(works_path, f"{work.id}.json")
            if os.path.exists(out_work):
                continue
            with open(out_work, "w", encoding="utf-8") as f:
                json.dump(work.to_dict(), f, ensure_ascii=False, indent=4)

    print("Indexing works")
    # index_authors(works_path, os.path.join(out_url, "indexed_works.json"))
    with open(os.path.join(out_url, "expert_works.json"), "w", encoding="utf-8") as f:
        json.dump(author_works, f, ensure_ascii=False, indent=4)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract openalex profiles"
    )
    parser.add_argument("info", help="Path to JSON info file")
    parser.add_argument("--out", required=True, help="output folder")

    args = parser.parse_args()

    process_works(args.info, args.out)


if __name__ == "__main__":
    main()

