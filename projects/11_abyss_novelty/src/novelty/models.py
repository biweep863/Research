"""Small torch models on log-mel clips, CPU-friendly.

* ``CNNClassifier``: 3 conv blocks -> global pool -> ``embed``-d embedding ->
  linear head.  Trained with cross-entropy on the *known* species only.  The
  embedding feeds the kNN / Mahalanobis novelty scores, the logits feed
  max-softmax and energy.
* ``ConvAE``: a convolutional autoencoder trained on known clips; the
  reconstruction error is a classifier-free novelty score.
* ``fit_pca_logistic``: the sklearn baseline (PCA + multinomial logistic).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))


class CNNClassifier(nn.Module):
    def __init__(self, n_classes: int, embed: int = 64):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64))
        self.embed = nn.Sequential(nn.Linear(64, embed), nn.ReLU())
        self.head = nn.Linear(embed, n_classes)

    def forward(self, x, return_embed: bool = False):
        h = self.features(x).mean(dim=(2, 3))
        z = self.embed(h)
        out = self.head(z)
        return (out, z) if return_embed else out


class ConvAE(nn.Module):
    def __init__(self, latent: int = 32, shape=(64, 96)):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU())
        h, w = shape[0] // 8, shape[1] // 8
        self.h, self.w = h, w
        self.fc1 = nn.Linear(32 * h * w, latent)
        self.fc2 = nn.Linear(latent, 32 * h * w)
        self.dec = nn.Sequential(nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                 nn.ReLU(), nn.ConvTranspose2d(16, 1, 4, 2, 1))

    def forward(self, x):
        z = self.fc1(self.enc(x).flatten(1))
        y = self.dec(self.fc2(z).view(-1, 32, self.h, self.w))
        return y


def _tensor(X: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(X, dtype=np.float32)).unsqueeze(1)


def train_classifier(X: np.ndarray, y: np.ndarray, n_classes: int, embed: int = 64, epochs: int = 18,
                     lr: float = 2e-3, batch: int = 64, seed: int = 0, device: str = "cpu") -> CNNClassifier:
    torch.manual_seed(seed)
    model = CNNClassifier(n_classes, embed).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt, yt = _tensor(X).to(device), torch.as_tensor(y, dtype=torch.long, device=device)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx] + 0.05 * torch.randn_like(Xt[idx])            # light augmentation
            loss = F.cross_entropy(model(xb), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def train_autoencoder(X: np.ndarray, latent: int = 32, epochs: int = 18, lr: float = 2e-3, batch: int = 64,
                      seed: int = 0, device: str = "cpu") -> ConvAE:
    torch.manual_seed(seed)
    model = ConvAE(latent, shape=X.shape[1:]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt = _tensor(X).to(device)
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]]
            loss = F.mse_loss(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


@torch.no_grad()
def logits_and_embed(model: CNNClassifier, X: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    out, z = model(_tensor(X).to(device), return_embed=True)
    return out.cpu().numpy(), z.cpu().numpy()


@torch.no_grad()
def recon_error(model: ConvAE, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    Xt = _tensor(X).to(device)
    return ((model(Xt) - Xt) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()


def fit_pca_logistic(X: np.ndarray, y: np.ndarray, dim: int = 40, seed: int = 0):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(PCA(n_components=min(dim, len(X) - 1, X[0].size), random_state=seed),
                         LogisticRegression(max_iter=2000, C=1.0))
    pipe.fit(X.reshape(len(X), -1), y)
    return pipe


def fit_probe(E_tr: np.ndarray, y_tr: np.ndarray, pca_dim: int = 128, C: float = 1.0, seed: int = 0):
    """Deterministic sklearn probe (StandardScaler -> PCA -> multinomial
    logistic regression) trained on frozen backbone embeddings. Used on the
    real-data path with a pretrained backbone in place of the CNN, since
    sklearn avoids the run-to-run CPU/GPU non-determinism noted in doc/03
    for from-scratch torch training."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    dim = min(pca_dim, len(E_tr) - 1, E_tr.shape[1])
    pipe = make_pipeline(StandardScaler(), PCA(n_components=dim, random_state=seed),
                         LogisticRegression(max_iter=5000, C=C))
    pipe.fit(E_tr, y_tr)
    return pipe
