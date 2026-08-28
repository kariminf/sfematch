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
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.prepare.split import multilabel_train_test_split



def main():
    parser = argparse.ArgumentParser(
        description="Multi-label-aware stratified train/test split."
    )
    parser.add_argument("input", help="Path to input .tsv")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction for the test set. Default: 0.2")
    parser.add_argument("--fields", default=None,
                        help="Comma-separated label columns. Default: "
                            "auto-detect all columns after 'keywords'.")
    parser.add_argument("--output-train", default=None)
    parser.add_argument("--output-test", default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-report", action="store_true",
                        help="Skip printing the representativeness report")
    args = parser.parse_args()

    fields = args.fields.split(",") if args.fields else None

    multilabel_train_test_split(
        args.input,
        test_size=args.test_size,
        fields=fields,
        output_train=args.output_train,
        output_test=args.output_test,
        random_state=args.random_state,
        report=not args.no_report,
    )


if __name__ == "__main__":
    main()