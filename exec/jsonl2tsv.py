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

from sfematch.prepare.format import json_to_tsv
from sfematch.prepare.arxiv_fields import *


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert an extracted arXiv JSON-lines file into a "
                    "tab-separated file: id, title, abstract, keywords, "
                    "and one binary (1/0) column per category field."
    )
    parser.add_argument("--src", required=True,
                        help="Path to input JSON-lines file")
    parser.add_argument("--out", default=None,
                        help="Path to output .tsv file. Default: same name "
                            "as input with .tsv extension")
    parser.add_argument("--dom", default="cs",
                            choices=["cs", "math", "stat", "bio", "fin", "eng", "phys"],
                            help="Predefined domain field set to use if "
                            "--fields is not given. Default: cs")

    args = parser.parse_args()

    fields = DOMAIN_FIELDS[args.dom]

    json_to_tsv(
        input_path=args.src,
        output=args.out,
        domain=args.dom,
    )


if __name__ == "__main__":
    main()


