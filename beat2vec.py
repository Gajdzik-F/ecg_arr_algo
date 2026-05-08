import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class BeatDatasetSSL(Dataset):
    def __init__(self, beats: np.ndarray):
        self.beats = beats.astype(np.float32)

    def __len__(self):
        return self.beats.shape[0]

    def __getitem__(self, idx):
        x = self.beats[idx]
        x1 = augment_beat(x)
        x2 = augment_beat(x)
        return x1[None, :], x2[None, :]  # [1, L]

def augment_beat(x):
    # x: [L]
    x = x.copy()
    L = len(x)

    # random amplitude scaling
    s = np.random.uniform(0.8, 1.2)
    x *= s

    # small time shift (circular roll)
    shift = np.random.randint(-int(0.02*L), int(0.02*L)+1)
    if shift != 0:
        x = np.roll(x, shift)

    # gaussian noise
    noise = np.random.normal(0, 0.03, size=L).astype(np.float32)
    x += noise

    # slight baseline drift perturbation (tiny sine)
    if np.random.rand() < 0.5:
        amp = np.random.uniform(0.0, 0.05)
        freq = np.random.uniform(0.5, 2.0)
        t = np.linspace(0, 1, L, endpoint=False)
        x += amp * np.sin(2*np.pi*freq*t).astype(np.float32)

    return x.astype(np.float32)

class Encoder1D(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, emb_dim)

    def forward(self, x):
        # x: [B,1,L]
        h = self.net(x).squeeze(-1)  # [B,64]
        z = self.fc(h)               # [B,emb_dim]
        z = nn.functional.normalize(z, dim=1)
        return z

def nt_xent_loss(z1, z2, temperature=0.2):
    """
    InfoNCE / NT-Xent (SimCLR) for a batch.
    z1,z2: [B,D] normalized
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # [2B,D]
    sim = torch.matmul(z, z.T) / temperature  # [2B,2B]
    # mask self-similarity
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+B
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]

    # denominator: logsumexp over row
    denom = torch.logsumexp(sim, dim=1)  # [2B]
    loss = - (pos - denom).mean()
    return loss

def train_beat2vec(beats, emb_dim=64, epochs=30, batch_size=256, lr=1e-3, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = BeatDatasetSSL(beats)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = Encoder1D(emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs+1):
        losses = []
        for x1, x2 in tqdm(dl, desc=f"SSL epoch {ep}/{epochs}", leave=False):
            x1 = x1.to(device)
            x2 = x2.to(device)
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2, temperature=0.2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[SSL] epoch {ep} loss={np.mean(losses):.4f}")

    return model

@torch.no_grad()
def embed_beats(model, beats, batch_size=512, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    x = torch.tensor(beats.astype(np.float32))[:, None, :]  # [N,1,L]
    embs = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i+batch_size].to(device)
        zb = model(xb).cpu().numpy()
        embs.append(zb)
    return np.vstack(embs)
