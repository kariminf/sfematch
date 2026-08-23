"""
Simple multilabel classifier: embedding -> MLP -> per-label logit -> sigmoid.
Trained with BCEWithLogitsLoss (numerically stabler than sigmoid + BCELoss).
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    """Wraps embedding + label arrays (numpy or memmap) for DataLoader use."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(np.asarray(self.X[idx], dtype=np.float32))
        y = torch.from_numpy(np.asarray(self.Y[idx], dtype=np.float32))
        return x, y


class MultilabelMLP(nn.Module):
    """embedding -> Linear -> ReLU -> Dropout -> Linear -> n_labels logits.
    No sigmoid inside the model -- BCEWithLogitsLoss expects raw logits and
    applies sigmoid internally for you, which is more numerically stable."""

    def __init__(self, input_dim, n_labels, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, x):
        return self.net(x)  # raw logits, shape (batch, n_labels)


def train_multilabel_model(
    X_train, Y_train, X_val=None, Y_val=None,
    hidden_dim=256, dropout=0.2, name="model",
    batch_size=512, epochs=30, lr=1e-3, device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    input_dim = X_train.shape[1]
    n_labels = Y_train.shape[1]

    model = MultilabelMLP(input_dim, n_labels, hidden_dim, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(EmbeddingDataset(X_train, Y_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(EmbeddingDataset(X_val, Y_val),
                                 batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        msg = f"{name}: epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = criterion(model(xb), yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)

    return model


def train_multilabel_model_2gpus(
    X_train, Y_train, X_val=None, Y_val=None,
    hidden_dim=256, dropout=0.2,
    batch_size=512, epochs=20, lr=1e-3,
    devices=("cuda:0", "cuda:1"),
):
    """Manual 2-GPU data-parallel training, single process (no torchrun/DDP needed).
 
    Per batch: split in half, one half goes to each GPU's own model replica,
    forward+backward run independently on each device, gradients are averaged
    together, the update is applied via one optimizer, then the other replica's
    weights are re-synced to match. Net effect is mathematically equivalent to
    training one model on the full batch, just computed across two GPUs.
 
    Note: the two forward/backward passes below are launched back-to-back
    *before* any blocking call (no .item()/.cpu() in between), so CUDA can run
    them concurrently on the two physical devices even though Python issues
    the instructions sequentially -- kernel launches are asynchronous per device.
    """
    assert torch.cuda.device_count() >= 2, "need at least 2 GPUs for this function"
    dev0, dev1 = devices
    input_dim = X_train.shape[1]
    n_labels = Y_train.shape[1]
 
    model0 = MultilabelMLP(input_dim, n_labels, hidden_dim, dropout).to(dev0)
    model1 = MultilabelMLP(input_dim, n_labels, hidden_dim, dropout).to(dev1)
    model1.load_state_dict(model0.state_dict())  # start identical
 
    optimizer0 = torch.optim.Adam(model0.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
 
    train_loader = DataLoader(EmbeddingDataset(X_train, Y_train),
                               batch_size=batch_size, shuffle=True)
    val_loader = None
    if X_val is not None:
        val_loader = DataLoader(EmbeddingDataset(X_val, Y_val),
                                 batch_size=batch_size, shuffle=False)
 
    for epoch in range(1, epochs + 1):
        model0.train()
        model1.train()
        total_loss = 0.0
        n_seen = 0
 
        for xb, yb in train_loader:
            half = xb.size(0) // 2
            if half == 0:
                # batch too small to split -- just run it on GPU 0 alone this step
                xb0, yb0 = xb.to(dev0), yb.to(dev0)
                xb1 = None
            else:
                xb0, yb0 = xb[:half].to(dev0, non_blocking=True), yb[:half].to(dev0, non_blocking=True)
                xb1, yb1 = xb[half:].to(dev1, non_blocking=True), yb[half:].to(dev1, non_blocking=True)
 
            optimizer0.zero_grad()
            model1.zero_grad()
 
            # launch both forward passes before any blocking call, so they can
            # actually overlap on the two GPUs
            logits0 = model0(xb0)
            logits1 = model1(xb1) if xb1 is not None else None
 
            loss0 = criterion(logits0, yb0)
            loss1 = criterion(logits1, yb1) if logits1 is not None else None
 
            loss0.backward()
            if loss1 is not None:
                loss1.backward()
 
            n_batch = xb0.size(0)
            batch_loss = loss0.item() * n_batch  # first .item() syncs dev0's stream here
 
            if loss1 is not None:
                n1 = xb1.size(0)
                batch_loss += loss1.item() * n1  # syncs dev1's stream here
                n_batch += n1
 
                # average model1's gradients (moved to dev0) into model0's gradients
                for p0, p1 in zip(model0.parameters(), model1.parameters()):
                    if p1.grad is not None:
                        g1 = p1.grad.to(dev0)
                        p0.grad = (p0.grad + g1) / 2 if p0.grad is not None else g1
 
            optimizer0.step()
 
            # keep model1 in sync with model0's just-updated weights
            model1.load_state_dict({k: v.to(dev1) for k, v in model0.state_dict().items()})
 
            total_loss += batch_loss
            n_seen += n_batch
 
        train_loss = total_loss / n_seen
        msg = f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}"
 
        if val_loader is not None:
            model0.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(dev0), yb.to(dev0)
                    loss = criterion(model0(xb), yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_loader.dataset)
            msg += f"  val_loss={val_loss:.4f}"
        print(msg)
 
    return model0  # model0 and model1 hold identical weights by this point
 

@torch.no_grad()
def predict_proba(model, X, batch_size=512, device=None):
    """Returns (n_samples, n_labels) numpy array of P(label=1), via sigmoid on logits."""
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(EmbeddingDataset(X, np.zeros((X.shape[0], 1))),  # dummy labels, unused
                         batch_size=batch_size, shuffle=False)
    outputs = []
    for xb, _ in loader:
        xb = xb.to(device)
        logits = model(xb)
        outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def save_model(model, path):
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": model.net[0].in_features,
        "n_labels": model.net[-1].out_features,
        "hidden_dim": model.net[0].out_features,
    }, path)
    print(f"saved model to {path}")


def load_model(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    model = MultilabelMLP(ckpt["input_dim"], ckpt["n_labels"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # smoke test with synthetic data
    np.random.seed(0)
    n, hidden, n_labels = 2000, 64, 5
    X = np.random.randn(n, hidden).astype(np.float32)
    Y = (np.random.rand(n, n_labels) > 0.7).astype(np.float32)

    n_train = int(n * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    model = train_multilabel_model(X_train, Y_train, X_val, Y_val, epochs=3)
    proba = predict_proba(model, X_val[:5])
    print("proba shape:", proba.shape)
    print(proba)

    save_model(model, "/tmp/multilabel_mlp.pt")
    loaded = load_model("/tmp/multilabel_mlp.pt")
    proba2 = predict_proba(loaded, X_val[:5])
    assert np.allclose(proba, proba2, atol=1e-5)
    print("save/load roundtrip OK")
