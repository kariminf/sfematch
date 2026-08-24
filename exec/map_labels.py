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

from sfematch.prepare.labels import generate_all_mappings, map_labels




def main():
    parser = argparse.ArgumentParser(
        description="Generate mapped labels "
                    "using fuzzy (possibly multi-target) mappings."
    )

    parser.add_argument("input", help="Path to input labels .tsv (id + source columns)")

    parser.add_argument("--map", default=None, help="Path to mapping file. Default: None")
    
    parser.add_argument("--output-ccs2012l1", default=None,
                        help="Output path for CCS2012 L1 labels. Default: "
                            "<input>_ccs2012l1.tsv")
    parser.add_argument("--output-ccsf", default=None,
                        help="Output path for Core CS Fields labels. "
                            "Default: <input>_ccsf.tsv")
    args = parser.parse_args()

    if args.map is None:
        generate_all_mappings(
            args.input,
            output_ccs2012l1=None,
            output_ccsf=None,
        )
    else:
        with open(args.map, encoding="utf-8") as f:
            map_info = json.load(f)

        # print(map_info)
        output_path = args.input.rsplit(".", 1)[0] + "_" + map_info["name"] + ".tsv"

        df_ccsf = map_labels(args.input, map_info["target_names"], map_info["mapping"], output_path)


if __name__ == "__main__":
    main()