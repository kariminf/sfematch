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
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sfematch.model.plm_encoding import Config, create_emb_fct, run_embedding



def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for a dataset (title+abstract) using an mlp."
    )
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config_json = json.load(f)

    config = Config()
    config.fill(config_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    embed_fct = create_emb_fct(config, device)

    run_embedding(config, embed_fct)


if __name__ == "__main__":
    main()