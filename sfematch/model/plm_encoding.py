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

# from __future__ import annotations

import os
import json
from typing import Callable, Optional, Tuple

from dataclasses import dataclass, fields, asdict

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch


def iter_titles_abstracts(tsv_path, batch_size=2000):
    reader = pd.read_csv(
        tsv_path, sep="\t", usecols=["id", "title", "abstract"],
        dtype=str, chunksize=batch_size, keep_default_na=False, quoting=3
    )
    for chunk in reader:
        titles = chunk["title"].fillna("").str.strip().tolist()
        abstracts = chunk["abstract"].fillna("").str.strip().tolist()
        yield titles, abstracts


def count_rows_safe(tsv_path, chunksize=100_000):
    total = 0
    for chunk in pd.read_csv(tsv_path, sep="\t", usecols=["id"], dtype=str, chunksize=chunksize, quoting=3):
        total += len(chunk)
    return total

@dataclass
class Config:
    tsv_path: str = ""
    out_name: str = "output"
    model_name: str = "bert"
    batch_size: int = 16
    max_length: int = 512
    read_batch_size: int = 3000
    flush_freq: int = 5000
    fp16: bool = False
    out_dir: str = "."
    resume: bool = False
    join_str: str = ". "
    hidden_size: int = 712

    def fill(self, obj: dict):
        for f in fields(self):
            if f.name in obj:
                setattr(self, f.name, obj[f.name])

    def to_dict(self) -> dict:
        return asdict(self)


def _load_standard(model_name: str):
    """Plain AutoModel-based PLM (BERT-style)."""
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def _load_specter2(model_name: str):
    """SPECTER2 needs the adapters library plus an explicit adapter load."""
    from transformers import AutoTokenizer
    from adapters import AutoAdapterModel

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoAdapterModel.from_pretrained(model_name)
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    return tokenizer, model


def _load_sbert(model_name: str, device: str):
    """SentenceTransformer manages its own device placement/eval mode,
    so it's handled separately rather than going through the shared
    to/eval/half logic below (SentenceTransformer objects don't expose
    the same .half()/.eval() contract as a raw transformers model)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    return None, model


# Registry: plm name -> (loader_fn, model_name, needs_device_arg)
# needs_device_arg distinguishes loaders (like sbert) that take device as
# a constructor arg from ones that get device applied afterward via .to().
_PLM_REGISTRY: dict[str, Tuple[Callable, str, bool]] = {
    "bert": (_load_standard, "bert-base-uncased", False),
    "scibert": (_load_standard, "allenai/scibert_scivocab_uncased", False),
    "specter2": (_load_specter2, "allenai/specter2_base", False),
    "sbert": (_load_sbert, "sentence-transformers/all-mpnet-base-v2", True),
}


def load_plm(plm: str, FP16: bool = False, device: str = "cpu"):
    if plm not in _PLM_REGISTRY:
        raise ValueError(f"Unknown plm '{plm}'. Choose from: {sorted(_PLM_REGISTRY)}")

    loader_fn, model_name, needs_device_arg = _PLM_REGISTRY[plm]

    if needs_device_arg:
        return loader_fn(model_name, device)

    tokenizer, model = loader_fn(model_name)
    model.to(device)
    model.eval()
    if FP16 and device == "cuda":
        model.half()

    return tokenizer, model

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def create_emb_fct(config: Config, device: str = "cpu"):
    tokenizer, model = load_plm(config.model_name, config.fp16, device)

    if config.model_name == "sbert":
        config.hidden_size = model.get_embedding_dimension()
        def sbert_embed_fct(texts):
            return model.encode(texts, batch_size=config.batch_size, show_progress_bar=False,
                                       convert_to_numpy=True).astype(np.float32)
        return sbert_embed_fct
    elif config.model_name == "specter2":
        config.hidden_size = model.config.hidden_size
        @torch.no_grad()
        def specter2_embed_fct(texts):
            tokens = tokenizer(
                texts, truncation=True, max_length=config.max_length,
                return_token_type_ids=False, padding=True, return_tensors="pt",
            ).to(device)
            output = model(**tokens)
            cls = output.last_hidden_state[:, 0, :]
            return cls.float().cpu().numpy()
        return specter2_embed_fct

    # bert and scibert
    config.hidden_size = model.config.hidden_size
    @torch.no_grad()
    def bert_family_embed_fct(texts):
        tokens = tokenizer(
            texts, truncation=True, max_length=config.max_length,
            padding=True, return_tensors="pt",
        ).to(device)
        output = model(**tokens)
        pooled = mean_pool(output.last_hidden_state, tokens["attention_mask"])
        return pooled.float().cpu().numpy()
    
    return bert_family_embed_fct

    

def join_title_abstract(title, abstract, join_str = ". "):
    return title + join_str + abstract

def checkpoint_path(out_path):
    return out_path + ".ckpt.json"


def load_checkpoint(out_path):
    path = checkpoint_path(out_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_checkpoint(out_path, row_idx, n_rows, hidden_size, model_name):
    with open(checkpoint_path(out_path), "w") as f:
        json.dump({"row_idx": row_idx, "n_rows": n_rows, "hidden_size": hidden_size, "model": model_name}, f)


def run_embedding(config: Config, embed_fn):
    out_path = os.path.join(config.out_dir, config.out_name)
    n_rows = count_rows_safe(config.tsv_path)
    print(f"[{config.out_name}] {n_rows:,} rows, hidden_size={config.hidden_size}", flush=True)

    ckpt = load_checkpoint(out_path) if config.resume else None
    start_row = 0

    if ckpt and ckpt["n_rows"] == n_rows and ckpt["hidden_size"] == config.hidden_size and ckpt["model"] == config.model_name:
        print(f"  resuming from row {ckpt['row_idx']:,}", flush=True)
        embeddings = np.lib.format.open_memmap(out_path, mode="r+", dtype=np.float32, shape=(n_rows, config.hidden_size))
        start_row = ckpt["row_idx"]
    else:
        if ckpt:
            print("  checkpoint found but doesn't match this run -- starting fresh.", flush=True)
        embeddings = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_rows, config.hidden_size))

    if start_row >= n_rows:
        print("  already complete.", flush=True)
        return

    row_idx = start_row
    rows_seen = 0
    last_flush = row_idx

    for titles, abstracts in tqdm(iter_titles_abstracts(config.tsv_path, config.read_batch_size),
                                   desc=config.out_name, total=(n_rows // config.read_batch_size) + 1):
        chunk_len = len(titles)
        chunk_start = rows_seen

        if chunk_start + chunk_len <= start_row:
            rows_seen += chunk_len
            continue

        for start in range(0, chunk_len, config.batch_size):
            batch_len = min(config.batch_size, chunk_len - start)
            t_batch = titles[start:start + batch_len]
            a_batch = abstracts[start:start + batch_len]
            texts = [join_title_abstract(t, a, join_str=config.join_str) for t, a in zip(t_batch, a_batch)]

            emb = embed_fn(texts)
            n = emb.shape[0]
            embeddings[row_idx:row_idx + n] = emb
            row_idx += n

        row_idx = chunk_start + chunk_len
        rows_seen += chunk_len

        if row_idx - last_flush >= config.flush_freq:
            embeddings.flush()
            save_checkpoint(out_path, row_idx, n_rows, config.hidden_size, config.model_name)
            last_flush = row_idx

    embeddings.flush()
    save_checkpoint(out_path, row_idx, n_rows, config.hidden_size, config.model_name)
    print(f"  done -> {out_path}", flush=True)
