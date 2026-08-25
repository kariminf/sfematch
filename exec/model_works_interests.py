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
import csv
import sys
from pathlib import Path

import numpy as np
import torch



# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def read_interests(path: Path):
    """Returns (ids: list[str], texts: list[str]) from interests.tsv."""
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            ids.append(row["id"])
            texts.append((row.get("keyword") or "").strip())
    return ids, texts


def read_works(path: Path):
    """Returns (ids: list[str], texts: list[str]) from works.tsv.
    Text fed to SBERT is 'title. abstract' (missing abstract handled gracefully)."""
    ids, texts = [], []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            text = f"{title}. {abstract}".strip(". ").strip()
            ids.append(row["id"])
            texts.append(text)
    return ids, texts


# --------------------------------------------------------------------------
# Encoding / classification
# --------------------------------------------------------------------------
def encode_texts(sbert_model, texts, batch_size=BATCH_SIZE):
    """SBERT-encode a list of strings -> (n, dim) float32 numpy array."""
    embeddings = sbert_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def run_classifiers(embeddings, device):
    """Runs every classifier in CLASSIFIERS on embeddings.
    Returns {name: (n, n_labels) float32 probability array}."""
    results = {}
    for name, model_path in CLASSIFIERS.items():
        if not model_path.exists():
            print(f"  [skip] {name}: model file not found at {model_path}")
            continue
        print(f"  running classifier: {name}")
        model = load_model(str(model_path), device=device)
        probs = predict_proba(model, embeddings, batch_size=BATCH_SIZE, device=device)
        results[name] = probs.astype(np.float32)
    return results


def process_split(name, ids, texts, sbert_model, device, out_dir: Path):
    """Encode + classify one split (interests or works) and save all .npy files."""
    print(f"\n=== {name} ===")
    print(f"{len(ids)} rows to encode")

    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings = encode_texts(sbert_model, texts)
    probs_by_classifier = run_classifiers(embeddings, device)

    np.save(out_dir / "ids.npy", np.array(ids, dtype=object))
    np.save(out_dir / "embeddings.npy", embeddings)
    for clf_name, probs in probs_by_classifier.items():
        np.save(out_dir / f"{clf_name}_probs.npy", probs)

    print(f"saved to {out_dir}/  "
          f"(ids.npy, embeddings.npy, {', '.join(k + '_probs.npy' for k in probs_by_classifier)})")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interests-tsv", type=Path, default=INTERESTS_TSV)
    parser.add_argument("--works-tsv", type=Path, default=WORKS_TSV)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sbert-model", type=str, default=SBERT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    global CLASSIFIERS, BATCH_SIZE
    BATCH_SIZE = args.batch_size
    CLASSIFIERS = {
        "arxiv": args.model_dir / "cs_arxiv_cls_sbert_mlmlp.pt",
        "ccs2012l1": args.model_dir / "cs_ccs2012l1_cls_sbert_mlmlp.pt",
        "ccsf": args.model_dir / "cs_ccsf_cls_sbert_mlmlp.pt",
    }

    if not args.interests_tsv.exists():
        sys.exit(f"interests tsv not found: {args.interests_tsv}")
    if not args.works_tsv.exists():
        sys.exit(f"works tsv not found: {args.works_tsv}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print(f"loading sentence-bert model: {args.sbert_model}")
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer(args.sbert_model, device=device)

    interest_ids, interest_texts = read_interests(args.interests_tsv)
    work_ids, work_texts = read_works(args.works_tsv)

    process_split("interests", interest_ids, interest_texts, sbert_model,
                   device, args.output_dir / "interests")
    process_split("works", work_ids, work_texts, sbert_model,
                   device, args.output_dir / "works")

    print("\ndone.")


if __name__ == "__main__":
    main()
