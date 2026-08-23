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

"""
Fine-tuned SentenceBERT (last N transformer blocks unfrozen) + MLP head,
trained end-to-end on title+abstract text with BCEWithLogitsLoss.

Pipeline: TSV -> title+abstract -> tokenize ONCE (no padding yet, truncation
to max_length) -> input_ids/attention_mask -> raw text discarded -> DataLoader
dynamically pads each batch to its own longest sequence (cheaper than always
padding to max_length). FP16 (autocast + GradScaler) is used throughout for
speed/memory on T4-class GPUs.
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def tokenize_dataset(titles, abstracts, tokenizer, max_length=256, chunk_size=5000):
    """Tokenizes ONCE, up front, in CHUNKS (bounds peak memory instead of
    handing the tokenizer all ~780K texts in one call), and packs every row's
    token ids into ONE flat int32 array + an `offsets` array (ragged/CSR-style
    layout) rather than one Python/torch tensor object per row -- avoids the
    heavy per-object overhead of ~1.5M separate small tensors, which is what
    was blowing up memory. attention_mask is NOT stored: before padding it's
    trivially all-ones, so it costs nothing to reconstruct at collate time
    instead of keeping a redundant full-size copy around."""
    n = len(titles)
    offsets = np.zeros(n + 1, dtype=np.int64)
    chunks_ids = []

    for start in tqdm(range(0, n, chunk_size), desc="tokenizing", unit="chunk"):
        end = min(start + chunk_size, n)
        texts = [f"{titles[i]}. {abstracts[i]}" if titles[i] else abstracts[i]
                 for i in range(start, end)]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        for j, ids in enumerate(enc["input_ids"]):
            offsets[start + j + 1] = offsets[start + j] + len(ids)
            chunks_ids.append(np.asarray(ids, dtype=np.int32))

    ids_flat = np.concatenate(chunks_ids)
    del chunks_ids
    return ids_flat, offsets


class TokenizedDataset(Dataset):
    """Holds tokens in packed form: one flat int32 array + offsets marking
    where each row's tokens start/end (offsets[i]:offsets[i+1]) -- a single
    slice + copy per __getitem__ call, no per-row object stored persistently."""

    def __init__(self, ids_flat, offsets, Y):
        self.ids_flat = ids_flat
        self.offsets = offsets
        self.Y = Y

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, idx):
        start, end = self.offsets[idx], self.offsets[idx + 1]
        ids = torch.from_numpy(self.ids_flat[start:end].astype(np.int64))
        y = torch.from_numpy(np.asarray(self.Y[idx], dtype=np.float32))
        return ids, y


def make_dynamic_pad_collate_fn(tokenizer):
    """Pads each batch to the longest sequence IN THAT BATCH, not to a fixed
    max_length every time. attention_mask is reconstructed here (all-ones for
    real tokens) since it was never stored -- cheap, one alloc per batch
    rather than one persistent array for the whole dataset."""
    def collate_fn(batch):
        ids_list, ys = zip(*batch)
        attn_list = [torch.ones(len(ids), dtype=torch.long) for ids in ids_list]
        padded = tokenizer.pad(
            {"input_ids": list(ids_list), "attention_mask": attn_list},
            padding=True, return_tensors="pt",
        )
        y = torch.stack(ys)
        return padded, y
    return collate_fn


class SBERTMultilabelClassifier(nn.Module):
    """SentenceBERT encoder (partially frozen) -> mean pooling -> MLP head.
    Only the last `n_unfrozen_layers` transformer blocks are trainable --
    embeddings and earlier layers stay frozen, which is cheaper and less
    prone to overfitting/catastrophic forgetting than full fine-tuning."""

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2",
                 n_labels=13, hidden_dim=256, dropout=0.2, n_unfrozen_layers=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        for p in self.encoder.parameters():
            p.requires_grad = False

        layers = self.encoder.encoder.layer  # ModuleList -- works for BERT-family incl. MPNet
        for layer in layers[-n_unfrozen_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(output.last_hidden_state, attention_mask)
        return self.head(pooled)  # raw logits

    def trainable_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_sbert_finetune(
    titles_train, abstracts_train, Y_train,
    titles_val=None, abstracts_val=None, Y_val=None,
    model_name="sentence-transformers/all-mpnet-base-v2",
    n_unfrozen_layers=4, hidden_dim=256, dropout=0.2,
    batch_size=32, epochs=3, encoder_lr=2e-5, head_lr=1e-3,
    max_length=256, device=None, name="sbert_ft", fp16=True,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = fp16 and device.startswith("cuda")
    print(f"{name}: training on device {device}  (fp16={use_fp16})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_labels = Y_train.shape[1]

    print(f"{name}: tokenizing {len(titles_train):,} train examples...")
    train_ids_flat, train_offsets = tokenize_dataset(titles_train, abstracts_train, tokenizer, max_length)
    val_ids_flat = val_offsets = None
    if titles_val is not None:
        print(f"{name}: tokenizing {len(titles_val):,} val examples...")
        val_ids_flat, val_offsets = tokenize_dataset(titles_val, abstracts_val, tokenizer, max_length)

    model = SBERTMultilabelClassifier(
        model_name=model_name, n_labels=n_labels, hidden_dim=hidden_dim,
        dropout=dropout, n_unfrozen_layers=n_unfrozen_layers,
    ).to(device)
    print(f"{name}: trainable params = {model.trainable_parameter_count():,}")

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": encoder_lr},
        {"params": model.head.parameters(), "lr": head_lr},
    ])
    criterion = nn.BCEWithLogitsLoss()
    collate_fn = make_dynamic_pad_collate_fn(tokenizer)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    train_loader = DataLoader(TokenizedDataset(train_ids_flat, train_offsets, Y_train),
                               batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_ids_flat is not None:
        val_loader = DataLoader(TokenizedDataset(val_ids_flat, val_offsets, Y_val),
                                 batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch}/{epochs}", unit="batch")
        for tokens, yb in pbar:
            tokens = {k: v.to(device, non_blocking=True) for k, v in tokens.items()}
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits = model(tokens["input_ids"], tokens["attention_mask"])
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * yb.size(0)
            n_seen += yb.size(0)
            pbar.set_postfix(loss=f"{total_loss / n_seen:.4f}")

        train_loss = total_loss / n_seen
        msg = f"{name}: epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            model.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for tokens, yb in val_loader:
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    yb = yb.to(device)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        loss = criterion(model(tokens["input_ids"], tokens["attention_mask"]), yb)
                    val_loss += loss.item() * yb.size(0)
                    n_val += yb.size(0)
            val_loss /= n_val
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

    return model, tokenizer


def train_sbert_finetune_2gpus(
    titles_train, abstracts_train, Y_train,
    titles_val=None, abstracts_val=None, Y_val=None,
    model_name="sentence-transformers/all-mpnet-base-v2",
    n_unfrozen_layers=4, hidden_dim=256, dropout=0.2,
    batch_size=32, epochs=3, encoder_lr=2e-5, head_lr=1e-3,
    max_length=256, devices=("cuda:0", "cuda:1"), name="sbert_ft_2gpu", fp16=True,
):
    """Manual 2-GPU data-parallel fine-tuning, single process. Only re-syncs
    TRAINABLE parameters (unfrozen blocks + head) between replicas each step
    -- frozen blocks are identical on both replicas from initialization and
    never change, so copying them every step would be wasted bandwidth.

    FP16: both losses are scaled by the SAME GradScaler instance before
    either .backward() call, so grad0 and grad1 end up scaled by the same
    factor -- averaging them stays numerically consistent, and only
    optimizer0 (which scaler.step() actually unscales) needs to be tracked."""
    assert torch.cuda.device_count() >= 2, "need at least 2 GPUs for this function"
    dev0, dev1 = devices
    use_fp16 = fp16 and torch.cuda.is_available()
    print(f"{name}: training on {dev0} + {dev1}  (fp16={use_fp16})")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n_labels = Y_train.shape[1]

    print(f"{name}: tokenizing {len(titles_train):,} train examples...")
    train_ids_flat, train_offsets = tokenize_dataset(titles_train, abstracts_train, tokenizer, max_length)
    val_ids_flat = val_offsets = None
    if titles_val is not None:
        print(f"{name}: tokenizing {len(titles_val):,} val examples...")
        val_ids_flat, val_offsets = tokenize_dataset(titles_val, abstracts_val, tokenizer, max_length)

    model0 = SBERTMultilabelClassifier(
        model_name=model_name, n_labels=n_labels, hidden_dim=hidden_dim,
        dropout=dropout, n_unfrozen_layers=n_unfrozen_layers,
    ).to(dev0)
    model1 = SBERTMultilabelClassifier(
        model_name=model_name, n_labels=n_labels, hidden_dim=hidden_dim,
        dropout=dropout, n_unfrozen_layers=n_unfrozen_layers,
    ).to(dev1)
    model1.load_state_dict(model0.state_dict())  # start identical
    print(f"{name}: trainable params per replica = {model0.trainable_parameter_count():,}")

    encoder_params0 = [p for p in model0.encoder.parameters() if p.requires_grad]
    optimizer0 = torch.optim.AdamW([
        {"params": encoder_params0, "lr": encoder_lr},
        {"params": model0.head.parameters(), "lr": head_lr},
    ])
    criterion = nn.BCEWithLogitsLoss()
    collate_fn = make_dynamic_pad_collate_fn(tokenizer)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    train_loader = DataLoader(TokenizedDataset(train_ids_flat, train_offsets, Y_train),
                               batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_ids_flat is not None:
        val_loader = DataLoader(TokenizedDataset(val_ids_flat, val_offsets, Y_val),
                                 batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for epoch in range(1, epochs + 1):
        model0.train()
        model1.train()
        total_loss, n_seen = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{name} epoch {epoch}/{epochs}", unit="batch")

        for tokens, yb in pbar:
            n_total = yb.size(0)
            half = n_total // 2

            if half == 0:
                tokens0 = {k: v.to(dev0) for k, v in tokens.items()}
                yb0 = yb.to(dev0)
                tokens1 = yb1 = None
            else:
                tokens0 = {k: v[:half].to(dev0, non_blocking=True) for k, v in tokens.items()}
                tokens1 = {k: v[half:].to(dev1, non_blocking=True) for k, v in tokens.items()}
                yb0 = yb[:half].to(dev0, non_blocking=True)
                yb1 = yb[half:].to(dev1, non_blocking=True)

            optimizer0.zero_grad()
            model1.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                logits0 = model0(tokens0["input_ids"], tokens0["attention_mask"])
                loss0 = criterion(logits0, yb0)
            logits1 = loss1 = None
            if tokens1 is not None:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    logits1 = model1(tokens1["input_ids"], tokens1["attention_mask"])
                    loss1 = criterion(logits1, yb1)

            scaler.scale(loss0).backward()
            if loss1 is not None:
                scaler.scale(loss1).backward()

            n_batch = yb0.size(0)
            batch_loss = loss0.item() * n_batch

            if loss1 is not None:
                n1 = yb1.size(0)
                batch_loss += loss1.item() * n1
                n_batch += n1

                # average gradients for TRAINABLE params only (still in
                # scaler-scaled units -- both losses were scaled by the same
                # factor, so averaging preserves that scaling consistently)
                for p0, p1 in zip(model0.parameters(), model1.parameters()):
                    if p0.requires_grad and p1.grad is not None:
                        g1 = p1.grad.to(dev0)
                        p0.grad = (p0.grad + g1) / 2 if p0.grad is not None else g1

            scaler.step(optimizer0)
            scaler.update()

            # re-sync only trainable params to model1 -- frozen ones never
            # diverge from their shared initialization, so skip them entirely
            with torch.no_grad():
                for p0, p1 in zip(model0.parameters(), model1.parameters()):
                    if p0.requires_grad:
                        p1.data.copy_(p0.data.to(dev1))

            total_loss += batch_loss
            n_seen += n_batch
            pbar.set_postfix(loss=f"{total_loss / n_seen:.4f}")

        train_loss = total_loss / n_seen
        msg = f"{name}: epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"

        if val_loader is not None:
            model0.eval()
            val_loss, n_val = 0.0, 0
            with torch.no_grad():
                for tokens, yb in val_loader:
                    tokens = {k: v.to(dev0) for k, v in tokens.items()}
                    yb = yb.to(dev0)
                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        loss = criterion(model0(tokens["input_ids"], tokens["attention_mask"]), yb)
                    val_loss += loss.item() * yb.size(0)
                    n_val += yb.size(0)
            val_loss /= n_val
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

    return model0, tokenizer  # model0 and model1 hold identical trainable weights by this point


@torch.no_grad()
def predict_proba_finetuned(model, tokenizer, titles, abstracts, batch_size=64,
                             max_length=256, device=None, fp16=True):
    device = device or next(model.parameters()).device
    use_fp16 = fp16 and str(device).startswith("cuda")
    model.eval()

    ids_flat, offsets = tokenize_dataset(titles, abstracts, tokenizer, max_length)
    dummy_y = np.zeros((len(titles), 1))  # unused, just satisfies TokenizedDataset's interface
    collate_fn = make_dynamic_pad_collate_fn(tokenizer)
    loader = DataLoader(TokenizedDataset(ids_flat, offsets, dummy_y),
                         batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    outputs = []
    for tokens, _ in tqdm(loader, desc="predicting", unit="batch"):
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.cuda.amp.autocast(enabled=use_fp16):
            logits = model(tokens["input_ids"], tokens["attention_mask"])
        outputs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def save_finetuned(model, path):
    """Saves the FULL model state (fine-tuned encoder + head), not just a
    small head like the frozen-embedding version -- expect ~few hundred MB,
    not a few MB."""
    torch.save({
        "state_dict": model.state_dict(),
        "n_labels": model.head[-1].out_features,
        "hidden_dim": model.head[0].out_features,
        "model_name": model.encoder.config._name_or_path,
    }, path)
    print(f"saved model to {path}")


def load_finetuned(path, n_unfrozen_layers=4, dropout=0.2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = SBERTMultilabelClassifier(
        model_name=ckpt["model_name"], n_labels=ckpt["n_labels"],
        hidden_dim=ckpt["hidden_dim"], dropout=dropout,
        n_unfrozen_layers=n_unfrozen_layers,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
    return model, tokenizer
