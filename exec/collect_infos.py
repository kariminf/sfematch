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

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.collect.openalex import get_ids as get_openalex_ids
from sfematch.collect.google_sholar import get_ids as get_gs_ids
from sfematch.collect.dblp import get_ids as get_dblp_ids
from sfematch.collect.orcid import get_ids as get_orcid_ids
from sfematch.collect.ieee import get_ids as get_ieee_ids
from sfematch.collect.sciencedirect import get_ids as get_sd_ids
from sfematch.collect.researchgate import get_ids as get_rg_ids
from sfematch.collect.other import get_github_ids, get_linkedin_ids, get_websites

MAX_IDS = 5
DEF_OUT = "./experts_info_choices.json"


def list_to_dict(lst):
    return [ac.to_dict() for ac in  lst]

def get_info(name: str, max_ids=5):

    print(f"Processing {name} ...")

    result = {}
    names = name.split(";")

    print("---- ORCID")
    result["orcid"] = []
    if len(names) > 1:
        result["orcid"] = list_to_dict(get_orcid_ids(names[0], names[1], max_results=max_ids))

    name = name.replace(";", " ")
    
    print("---- OpenAlex")
    result["openalex"] = list_to_dict(get_openalex_ids(name, per_page=max_ids))

    print("---- Google Scholar")
    result["google_scholar"] = list_to_dict(get_gs_ids(name, max_results=max_ids))

    print("---- ResearchGate")
    result["researchgate"] = list_to_dict(get_rg_ids(name, max_results=max_ids))

    print("---- DBLP")
    result["dblp"]     = list_to_dict(get_dblp_ids(name, max_results=max_ids))

    print("---- Science Direct")
    result["sciencedirect"]     = list_to_dict(get_sd_ids(name, max_results=max_ids))

    print("---- IEEE")
    result["ieeexplore"]     = list_to_dict(get_ieee_ids(name, max_results=max_ids))

    print("---- LinkedIn")
    result["linkedin"]     = list_to_dict(get_linkedin_ids(name, max_results=max_ids))

    print("---- Github")
    result["github"]     = list_to_dict(get_github_ids(name, max_results=max_ids))

    print("---- Website")
    result["website"]     = list_to_dict(get_websites(name, max_results=max_ids))
    


    return result


def get_infos(names):
    infos = []
    for name in names:
        infos.append(get_info(name))
    return infos


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('efile') 
    parser.add_argument("--out", type=str, default=DEF_OUT)
    parser.add_argument("--max-ids", type=int, default=MAX_IDS)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.efile):
        print(f"ERROR! File {args.efile} not found")
        sys.exit(1)

    infos = {}

    with open(args.efile, encoding="utf8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#"):
                continue
            infos[l] = get_info(l, max_ids=args.max_ids)


    with open(args.out, "w", encoding="utf8") as f:
        json.dump(infos, f, ensure_ascii=True, indent=4)

    print("\ndone.")



if __name__ == "__main__":
    main()


