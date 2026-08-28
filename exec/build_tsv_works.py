
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
import re
import sys

from deep_translator import GoogleTranslator
from langdetect import detect_langs
from pathlib import Path

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from sfematch.collect.openalex import index_authors


# OPENALEX_DIR = Path("./OpenAlex/Works")
# OWORKS_DIR = Path("./OtherWorks")
# EXPERTS_DIR = Path("./Experts")
# OUT_WORKS_DIR = Path("./ExpertsModeling")

IDW = 0
DEFAULT_LANG = "en"
PROB_TH = 0.90

translator = GoogleTranslator(source="auto", target=DEFAULT_LANG)


def clean_tsv_text(text):
    return re.sub(r"\s+", " ", text).strip()

def detect_language(text, src_languages=None):
    lang = DEFAULT_LANG
    try:
        results = detect_langs(text)
        if results[0].lang != DEFAULT_LANG and results[0].prob > PROB_TH:
            lang = results[0].lang
    except:
        lang = DEFAULT_LANG
    if src_languages is not None and lang not in src_languages:
        return DEFAULT_LANG
    return lang

def to_english(text, langs):

    lang = detect_language(text, src_languages=langs)

    if lang != DEFAULT_LANG:
        try:
            text = translator.translate(text)
            print("translating from", lang)
        except:
            print("cannot translate from", lang)

    return text


def openalex_short_id(openalex_id: str) -> str:
    """'https://openalex.org/W657975173' -> 'W657975173'"""
    return openalex_id.rstrip("/").split("/")[-1]

def prepare_work(data, langs):
    global IDW
    IDW += 1
    id = f"WRK{IDW:05d}"

    title    =  clean_tsv_text(data["title"])
    abstract =  clean_tsv_text(data["abstract"])

    if langs:
        title = to_english(title, langs)
        abstract = to_english(abstract, langs)

    return id, f"{id}\t{title}\t{abstract}\n"


# def process_many_works_per_json(data):
#     tsv = ""
#     id_map = {}
#     for old_id in data:
#         print(f"processing {old_id} ...")
#         work = data[old_id]
#         new_id, tsv_line = prepare_work(work)
#         id_map[old_id] = new_id
#         tsv += tsv_line
#     return id_map, tsv


# def process_other_works():
#     id_map_global, tsv_global = {}, ""
#     for pub_file in OWORKS_DIR.glob("*.json"):
#         with open(pub_file, encoding="utf-8") as f:
#             works = json.load(f)
#         id_map, tsv = process_many_works_per_json(works)
#         id_map_global.update(id_map)
#         tsv_global += tsv
#     return id_map_global, tsv_global

def process_openalex_works(openalex_dir: Path, langs):
    id_map_global, tsv_global = {}, ""
    for pub_file in openalex_dir.glob("*.json"):
        with open(pub_file, encoding="utf-8") as f:
            work = json.load(f)
        new_id, tsv = prepare_work(work, langs)

        old_id = openalex_short_id(work.get("id", pub_file.stem))
        print(f"processing {old_id} ...")

        id_map_global[old_id] = new_id
        tsv_global += tsv
    return id_map_global, tsv_global

def format_index(index: dict[str, list[dict]]) -> str:
    """
    Pretty-print the index, but keep each {"id":..,"pos":..,"nbr":..}
    entry on a single line instead of one line per key.
    """
    lines = ["{"]
    items = list(index.items())

    for i, (name, works) in enumerate(items):
        lines.append(f"  {json.dumps(name, ensure_ascii=False)}: [")
        for j, w in enumerate(works):
            comma = "," if j < len(works) - 1 else ""
            lines.append(f"    {json.dumps(w, ensure_ascii=False)}{comma}")
        comma = "," if i < len(items) - 1 else ""
        lines.append(f"  ]{comma}")

    lines.append("}")
    return "\n".join(lines)

def annonymize_expert_works(main_path, id_map):
    with open(main_path / "expert_works.json", encoding="utf-8") as f:
        expert_works = json.load(f)

    for eid in expert_works:
        for work in expert_works[eid]:
            if work["id"] in id_map:
                work["id"] = id_map[work["id"]]

    with open(main_path / "expert_works_anonym.json", "w", encoding="utf-8") as f:
        f.write(format_index(expert_works))
        # json.dump(expert_works, f, ensure_ascii=False, indent=2)



def process_works(main_path: Path, langs):

    id_map_final = {}
    tsv_final = "id\ttitle\tabstract\n"

    # id_map, tsv = process_other_works()
    # id_map_final.update(id_map)
    # tsv_final += tsv

    id_map, tsv = process_openalex_works(main_path / "works", langs)
    id_map_final.update(id_map)
    tsv_final += tsv

    with open(main_path / "works_id_mapping.json", "w", encoding="utf-8") as f:
        json.dump(id_map_final, f, ensure_ascii=False, indent=2)

    with open(main_path / "works.tsv", "w", encoding="utf-8") as f:
        f.write(tsv_final)

    annonymize_expert_works(main_path, id_map_final)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract openalex profiles"
    )
    parser.add_argument("path", help="Path to the main folder")
    parser.add_argument("--t", default=None, help="translate from; specify languages codes separated by ,")

    args = parser.parse_args()

    langs = None
    if args.t:
        langs = args.t.split(",")

    process_works(Path(args.path), langs)


if __name__ == "__main__":
    main()
