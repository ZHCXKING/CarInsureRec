import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
class MIWAENet(nn.Module):
    def __init__(self, x_dim, h_dim=128, z_dim=20):
        super(MIWAENet, self).__init__()
        self.enc_h1 = nn.Linear(x_dim, h_dim)
        self.enc_h2 = nn.Linear(h_dim, h_dim)
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_logvar = nn.Linear(h_dim, z_dim)
        self.dec_h1 = nn.Linear(z_dim, h_dim)
        self.dec_h2 = nn.Linear(h_dim, h_dim)
        self.dec_mu = nn.Linear(h_dim, x_dim)
        self.dec_logvar = nn.Linear(h_dim, x_dim)
    def encode(self, x):
        h = torch.tanh(self.enc_h1(x))
        h = torch.tanh(self.enc_h2(h))
        return self.enc_mu(h), self.enc_logvar(h)
    def decode(self, z):
        h = torch.tanh(self.dec_h1(z))
        h = torch.tanh(self.dec_h2(h))
        return self.dec_mu(h), self.dec_logvar(h)
    def forward(self, x, K=1):
        batch_size, x_dim = x.shape
        mu_q, logvar_q = self.encode(x)
        std_q = torch.exp(0.5 * logvar_q)
        eps = torch.randn(K, batch_size, mu_q.shape[1], device=x.device)
        z = mu_q.unsqueeze(0) + eps * std_q.unsqueeze(0)
        z_flat = z.view(-1, z.shape[-1])
        mu_p_flat, logvar_p_flat = self.decode(z_flat)
        mu_p = mu_p_flat.view(K, batch_size, x_dim)
        logvar_p = logvar_p_flat.view(K, batch_size, x_dim)
        return mu_q, logvar_q, z, mu_p, logvar_p
class MIWAEImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_hidden=128,
        n_latent=20,
        K=20,
        L=1000,
        batch_size=64,
        epoch=500,
        learning_rate=1e-3,
        seed=42,
        verbose=True,
    ):
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.K = K
        self.L = L
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.seed = seed
        self.verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    def _compute_loss(self, x, m, mu_q, logvar_q, z, mu_p, logvar_p):
        K = z.shape[0]
        x_rep = x.unsqueeze(0).expand(K, -1, -1)
        m_rep = m.unsqueeze(0).expand(K, -1, -1)
        std_p = torch.exp(0.5 * logvar_p)
        log_p_x_given_z = Normal(mu_p, std_p).log_prob(x_rep)
        log_p_x_given_z = (log_p_x_given_z * m_rep).sum(dim=-1)
        log_p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
        std_q = torch.exp(0.5 * logvar_q)
        log_q_z_given_x = Normal(mu_q, std_q).log_prob(z).sum(dim=-1)
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_mean_w = torch.logsumexp(log_w, dim=0) - np.log(K)
        loss = -torch.mean(log_mean_w)
        return loss
    def fit(self, X, y=None):
        self._set_seed()
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        X_np = X.values.astype(np.float32)
        mask_np = 1 - np.isnan(X_np).astype(np.float32)
        X_filled_temp = np.nan_to_num(X_np, nan=np.nanmean(X_np, axis=0))
        X_scaled = self.scaler.fit_transform(X_filled_temp)
        X_scaled[mask_np == 0] = 0.0
        X_t = torch.tensor(X_scaled, device=self.device)
        M_t = torch.tensor(mask_np, device=self.device)
        dataset = TensorDataset(X_t, M_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model = MIWAENet(
            x_dim=self.n_features_in_,
            h_dim=self.n_hidden,
            z_dim=self.n_latent
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        iterator = range(self.epoch)
        if self.verbose:
            iterator = tqdm(iterator, desc="MIWAE Training")
        self.model.train()
        for _ in iterator:
            for x_batch, m_batch in loader:
                optimizer.zero_grad()
                mu_q, logvar_q, z, mu_p, logvar_p = self.model(x_batch, K=self.K)
                loss = self._compute_loss(x_batch, m_batch, mu_q, logvar_q, z, mu_p, logvar_p)
                loss.backward()
                optimizer.step()
        self.is_fitted_ = True
        return self
    def transform(self, X):
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("MIWAEImputer not fitted.")
        self.model.eval()
        X_np = X.values.astype(np.float32)
        mask_np = 1 - np.isnan(X_np).astype(np.float32)
        X_filled_temp = np.nan_to_num(X_np, nan=self.scaler.mean_)
        X_scaled = self.scaler.transform(X_filled_temp)
        X_scaled[mask_np == 0] = 0.0
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        tensor_m = torch.tensor(mask_np, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_m)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        imputed_batches = []
        with torch.no_grad():
            for x_batch, m_batch in loader:
                x_batch = x_batch.to(self.device)
                m_batch = m_batch.to(self.device)
                mu_q, logvar_q, z, mu_p, logvar_p = self.model(x_batch, K=self.L)
                std_p = torch.exp(0.5 * logvar_p)
                x_exp = x_batch.unsqueeze(0).expand(self.L, -1, -1)
                m_exp = m_batch.unsqueeze(0).expand(self.L, -1, -1)
                log_p_x_given_z = Normal(mu_p, std_p).log_prob(x_exp)
                log_p_x_given_z = (log_p_x_given_z * m_exp).sum(dim=-1)
                log_p_z = Normal(0, 1).log_prob(z).sum(dim=-1)
                std_q = torch.exp(0.5 * logvar_q)
                log_q_z_given_x = Normal(mu_q, std_q).log_prob(z).sum(dim=-1)
                log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
                w = F.softmax(log_w, dim=0)
                x_imputed_batch = (w.unsqueeze(-1) * mu_p).sum(dim=0)
                imputed_batches.append(x_imputed_batch.cpu().numpy())
        x_imputed_scaled = np.concatenate(imputed_batches, axis=0)
        x_imputed = self.scaler.inverse_transform(x_imputed_scaled)
        x_final = X_np.copy()
        x_final[mask_np == 0] = x_imputed[mask_np == 0]
        return x_final
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_