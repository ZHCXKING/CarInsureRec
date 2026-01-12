# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
# %%
class Generator(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)
    def forward(self, x, m):
        inp = torch.cat([x, m], dim=1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))
class Discriminator(nn.Module):
    def __init__(self, dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, dim)

    def forward(self, x, h):
        inp = torch.cat([x, h], dim=1)
        h = F.relu(self.fc1(inp))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))
# %%
class GAINImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        batch_size=256,
        hint_rate=0.9,
        alpha=100,
        epoch=500,
        learning_rate=1e-3,
        seed=42,
        verbose=True,
    ):
        self.batch_size = batch_size
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.scaler = MinMaxScaler()
    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    def _sample_B(self, m, n):
        return np.random.binomial(1, self.hint_rate, size=(m, n))
    def fit(self, X, y=None):
        self._set_seed()
        X_np = X.values.astype(np.float32)
        mask_np = 1 - np.isnan(X_np).astype(np.float32)
        X_scaled = self.scaler.fit_transform(
            np.nan_to_num(X_np, nan=0.0)
        )
        X_t = torch.tensor(X_scaled, device=self.device)
        M_t = torch.tensor(mask_np, device=self.device)
        dataset = TensorDataset(X_t, M_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        n, dim = X_t.shape
        h_dim = dim
        self.G = Generator(dim, h_dim).to(self.device)
        self.D = Discriminator(dim, h_dim).to(self.device)
        opt_G = torch.optim.Adam(self.G.parameters(), lr=self.learning_rate)
        opt_D = torch.optim.Adam(self.D.parameters(), lr=self.learning_rate)
        iterator = range(self.epoch)
        if self.verbose:
            iterator = tqdm(iterator, desc="GAIN Training")
        for _ in iterator:
            for x, m in loader:
                z = torch.rand_like(x)
                b = torch.tensor(
                    self._sample_B(x.size(0), dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                h = b * m + 0.5 * (1 - b)
                x_tilde = m * x + (1 - m) * z
                with torch.no_grad():
                    x_bar = self.G(x_tilde, m)
                    x_hat = m * x + (1 - m) * x_bar
                d_prob = self.D(x_hat, h)
                d_loss = -torch.sum(
                    (b == 0)
                    * (
                        m * torch.log(d_prob + 1e-8)
                        + (1 - m) * torch.log(1 - d_prob + 1e-8)
                    )
                ) / torch.sum(b == 0)
                opt_D.zero_grad()
                d_loss.backward()
                opt_D.step()
                z = torch.rand_like(x)
                b = torch.tensor(
                    self._sample_B(x.size(0), dim),
                    device=self.device,
                    dtype=torch.float32,
                )
                h = b * m + 0.5 * (1 - b)
                x_tilde = m * x + (1 - m) * z
                x_bar = self.G(x_tilde, m)
                x_hat = m * x + (1 - m) * x_bar
                d_prob = self.D(x_hat, h)
                g_adv = -torch.sum(
                    (b == 0) * (1 - m) * torch.log(d_prob + 1e-8)
                ) / torch.sum((b == 0) * (1 - m))
                g_mse = torch.sum(
                    m * (x - x_bar) ** 2
                ) / torch.sum(m)
                g_loss = g_adv + self.alpha * g_mse
                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()
        return self
    def transform(self, X):
        self.G.eval()
        X_np = X.values.astype(np.float32)
        mask_np = 1 - np.isnan(X_np).astype(np.float32)
        X_scaled = self.scaler.transform(
            np.nan_to_num(X_np, nan=0.0)
        )
        x = torch.tensor(X_scaled, device=self.device)
        m = torch.tensor(mask_np, device=self.device)
        with torch.no_grad():
            z = torch.rand_like(x)
            x_tilde = m * x + (1 - m) * z
            x_bar = self.G(x_tilde, m)
            x_hat = m * x + (1 - m) * x_bar
        x_hat = x_hat.cpu().numpy()
        x_final = self.scaler.inverse_transform(x_hat)
        return x_final
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self.feature_names_in_)
        return np.asarray(input_features)
