
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
import argparse
from pathlib import Path
import json



EXPERT_ID_RE = re.compile(r'^XPRT\d+$', re.IGNORECASE)

IDI = 0



def parse_file(path: Path):

    global IDI

    expert_interests = {}

    keywords = {}


    with open(path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.rstrip('\n').rstrip('\r')
            stripped = line.strip()

            if not stripped:
                continue

            current_expert = None
            if EXPERT_ID_RE.match(stripped):
                current_expert = stripped
                expert_interests[current_expert] = set()
                continue

            if current_expert is None:
                # Line encountered before any expert ID; skip it.
                continue

            if ':' not in stripped:
                # Not a "type: interests" line; skip.
                continue

            type_name, _, value = stripped.partition(':')
            type_name = type_name.strip()
            value = value.strip()

            if not type_name:
                continue


            if value != '':
                for interest in value.split(';'):
                    interest = interest.strip()
                    if not interest:
                        continue
                    if interest in keywords:
                        int_id = keywords[interest]
                    else:
                        IDI += 1
                        int_id = f"INT{IDI:05d}"
                        keywords[interest] = int_id

                    expert_interests[current_expert].add(int_id)

    return expert_interests, keywords


def save_interests(path, data):

    tsv = "id\tkeyword\n"

    # order by v
    for k, v in sorted(data.items(), key=lambda x: x[1]):
        tsv += f"{v}\t{k}\n"

    with open(path / "interests.tsv", "w", encoding="utf-8") as f:
        f.write(tsv)

def save_experts_interests(path, data):
    data = {k: sorted(v) for k, v in data.items()}

    with open(path / "experts_interests.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def process_interests(int_path: Path, path: Path):

    expert_interests, keywords = parse_file(int_path)

    save_interests(path, keywords)
    save_experts_interests(path, expert_interests)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract openalex profiles"
    )
    parser.add_argument("path", help="Path to the main folder")
    parser.add_argument("--int", help="Interests file")

    args = parser.parse_args()


    process_interests(Path(args.int), Path(args.path))


if __name__ == "__main__":
    main()
