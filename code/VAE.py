from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import numpy as np
import random
import torch.nn.functional as F

class PriorEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(min=-10, max=10)
        return mu, logvar

def train_prior_encoder(data, input_dim, latent_dim, epochs, lr=1e-3):
    prior_encoder = PriorEncoder(input_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(prior_encoder.parameters(), lr=lr)
    
    # Add a decoder to reconstruct the original data
    decoder = nn.Sequential(
        nn.Linear(latent_dim, 512),
        nn.ReLU(),
        nn.Linear(512, input_dim)
    ).to(device)
    
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in loader:
            x = batch[0]
            mu, logvar = prior_encoder(x)
            
            # Use decoder for reconstruction
            reconstructed = decoder(mu)
            recon_loss = F.mse_loss(reconstructed, x)
            
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            loss = recon_loss + 0.1 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        print(f"prior total_loss: {total_loss}")
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch}, Prior Loss {avg_loss:.6f}")
    
    return prior_encoder

def vae_decoder(latent_dim, output_dim, hidden_dim, final_activation=None):
    layers = [
        nn.Linear(latent_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.BatchNorm1d(hidden_dim * 2),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim * 2, hidden_dim * 4),
        nn.BatchNorm1d(hidden_dim * 4),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_dim * 4, output_dim)
    ]
    if final_activation:
        layers.append(final_activation)
    return nn.Sequential(*layers)

class VAE(nn.Module):
    def __init__(self, input_dim, sga_dim, protein_dim, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder_rna = vae_decoder(latent_dim, input_dim, hidden_dim)
        self.decoder_sga = vae_decoder(latent_dim, sga_dim, hidden_dim)
        self.decoder_protein = vae_decoder(latent_dim, protein_dim, hidden_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(min=-10, max=10)
        std = torch.exp(0.5 * logvar).clamp(min=1e-6)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        rna_out = self.decoder_rna(z)
        sga_out = self.decoder_sga(z)
        protein_out = self.decoder_protein(z)
        return rna_out, sga_out, protein_out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rna_out, sga_out, protein_out = self.decode(z)
        return rna_out, sga_out, protein_out, mu, logvar

def vae_kl_divergence(mu, logvar, prior_mu, prior_logvar):
    kl = 0.5 * (prior_logvar - logvar + 
                (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp() - 1)
    return torch.mean(kl)

def vae_loss_function(rna_out, sga_out, protein_out, rna, sga, protein, mu, logvar, mean_mu_sga, mean_logvar_sga, mean_mu_protein, mean_logvar_protein):
    loss_rna = F.mse_loss(rna_out, rna, reduction='mean')
    loss_sga = F.mse_loss(sga_out, sga, reduction='mean')
    loss_protein = F.mse_loss(protein_out, protein, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    kl_sga = vae_kl_divergence(mu, logvar, mean_mu_sga, mean_logvar_sga)
    kl_protein = vae_kl_divergence(mu, logvar, mean_mu_protein, mean_logvar_protein)
    return loss_rna + loss_sga + loss_protein + kl_sga + kl_protein + kld

seed = 6889
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rna_data = pd.read_csv("../data/source/gdsc_rna_filter_data.csv")
sga_data = pd.read_csv("../data/source/gdsc_sga_filter_data.csv")
protein_data = pd.read_csv("../data/source/gdsc_protein_filter_data.csv")

rna = rna_data.set_index("cell_line_name")
sga = sga_data.set_index("cell_line_name")
protein = protein_data.set_index("cell_line_name")

common_samples = list(set(rna.index) & set(sga.index) & set(protein.index))
rna = rna.loc[common_samples]
sga = sga.loc[common_samples]
protein = protein.loc[common_samples]

scaler_rna = StandardScaler()
scaler_sga = StandardScaler()
scaler_protein = StandardScaler()
rna_std = scaler_rna.fit_transform(rna)
sga_std = scaler_sga.fit_transform(sga)
protein_std = scaler_protein.fit_transform(protein)

X_rna = torch.tensor(rna_std, dtype=torch.float32)
Y_sga = torch.tensor(sga_std, dtype=torch.float32)
Y_protein = torch.tensor(protein_std, dtype=torch.float32)

train_idx, test_idx = train_test_split(np.arange(X_rna.shape[0]), test_size=0.2, random_state=seed)
X_rna_train, X_rna_test = X_rna[train_idx], X_rna[test_idx]
Y_sga_train, Y_sga_test = Y_sga[train_idx], Y_sga[test_idx]
Y_protein_train, Y_protein_test = Y_protein[train_idx], Y_protein[test_idx]

X_rna_train = X_rna_train.to(device)
Y_sga_train = Y_sga_train.to(device)
Y_protein_train = Y_protein_train.to(device)
X_rna_test = X_rna_test.to(device)



batch_size = 64
epochs = 500
hidden_dim = 256
latent_dim = 64
protein_epochs = 1000
sga_epochs = 500
protein_lr = 1e-4
sga_lr = 1e-3

protein_prior = train_prior_encoder(Y_protein_train, Y_protein_train.shape[1], latent_dim, protein_epochs, protein_lr)
sga_prior = train_prior_encoder(Y_sga_train, Y_sga_train.shape[1], latent_dim, sga_epochs, sga_lr)

with torch.no_grad():
    mu_protein, logvar_protein = protein_prior(Y_protein_train)
    mu_sga, logvar_sga = sga_prior(Y_sga_train)
    mean_mu_protein = mu_protein.mean(dim=0)
    mean_logvar_protein = logvar_protein.mean(dim=0)
    mean_mu_sga = mu_sga.mean(dim=0)
    mean_logvar_sga = logvar_sga.mean(dim=0)


train_dataset = TensorDataset(X_rna_train, Y_sga_train, Y_protein_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = VAE(X_rna.shape[1], Y_sga.shape[1], Y_protein.shape[1], latent_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, epochs + 1):
    total_loss = 0
    model.train()
    for x_batch, sga_batch, protein_batch in train_loader:
        rna_out, sga_out, protein_out, mu, logvar = model(x_batch)
        loss = vae_loss_function(rna_out, sga_out, protein_out, x_batch, sga_batch, protein_batch, mu, logvar, mean_mu_sga, mean_logvar_sga, mean_mu_protein, mean_logvar_protein)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss {total_loss:.4f}")

torch.save(model.state_dict(), "../model/vae_model.pth")
