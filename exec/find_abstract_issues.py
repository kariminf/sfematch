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

import json
from pathlib import Path
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SRC_DIR = Path("./Works")   # folder with one JSON file per work
OUT_FILE = Path("./abstract_report.json")
SIMILARITY_THRESHOLD = 0.9


def load_works(src_dir: Path) -> list[dict]:
    works = []
    for path in src_dir.glob("*.json"):
        print(f"loading {path}")
        with open(path, encoding="utf-8") as f:
            work = json.load(f)
        work["_file"] = path.name
        works.append(work)
    return works


def find_missing_abstracts(works: list[dict]) -> list[str]:
    return [w.get("id", w["_file"]) for w in works if len((w.get("abstract") or "").strip()) < 300]


def find_similar_pairs(works: list[dict], threshold: float) -> list[dict]:
    # Only consider works that actually have an abstract.
    # candidates = [w for w in works if len((w.get("abstract") or "").strip()) >= 300]
    candidates = works
    if len(candidates) < 2:
        return []

    ids = [w.get("id", w["_file"]) for w in candidates]
    # abstracts = [w["abstract"] for w in candidates]
    texts = []
    for w in candidates:
        texts.append(w["title"] + " " + w["abstract"])

    vectorizer = CountVectorizer()          # plain term-frequency, not TF-IDF
    tf_matrix = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tf_matrix)

    pairs = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            score = sim_matrix[i, j]
            if score > threshold:
                pairs.append({
                    "id1": ids[i],
                    "id2": ids[j],
                    "cosine": round(float(score), 4),
                })

    pairs.sort(key=lambda p: p["cosine"], reverse=True)
    return pairs


def process_works(main_dir: Path):
    src_dir = main_dir / "works"
    works = load_works(src_dir)
    print(f"Loaded {len(works)} works from {src_dir}")

    missing = find_missing_abstracts(works)
    print(f"Works with missing/empty abstract: {len(missing)}")

    similar_pairs = find_similar_pairs(works, SIMILARITY_THRESHOLD)
    print(f"Similar pairs found (cosine > {SIMILARITY_THRESHOLD}): {len(similar_pairs)}")

    report = {
        "missing_abstract": missing,
        "similar_pairs": similar_pairs,
    }

    out_file = main_dir / "abstract_issues_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print(f"Saved report to {out_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract openalex profiles"
    )
    parser.add_argument("path", help="Path to the main folder containing the folder 'works'")

    args = parser.parse_args()

    process_works(Path(args.path))


if __name__ == "__main__":
    main()
