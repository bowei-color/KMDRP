

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import random
from transformers import RobertaTokenizerFast
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_ecfp_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate ECFP fingerprint from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # If SMILES is invalid, return zero vector
        return np.zeros(n_bits, dtype=np.float32)
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def generate_fingerprint_features(smiles_df):
    """Generate ECFP fingerprints for all SMILES in the dataframe"""
    fingerprints = []
    valid_cids = []
    
    for _, row in smiles_df.iterrows():
        pubchem_cid = row['pubchem_cid']
        smiles = row['smiles']
        
        fp = generate_ecfp_fingerprint(smiles)
        fingerprints.append(fp)
        valid_cids.append(pubchem_cid)
    
    # Create DataFrame with fingerprints
    fingerprint_df = pd.DataFrame(fingerprints)
    fingerprint_df.insert(0, 'pubchem_cid', valid_cids)
    
    return fingerprint_df


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
        logvar = self.fc_logvar(h).clamp(min=-10, max=10)
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


def make_mlp(input_dim, output_dim, hidden_dims, dropout=0.3, activation=nn.LeakyReLU(0.2)):
    layers = []
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation)
        layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class DrugEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, hidden_dim=256, num_layers=2, max_len=128):
        super(DrugEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x + self.position_embedding[:, :x.size(1), :]
        src_key_padding_mask = ~attention_mask.bool()
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x)

    def encode_embedding(self, input_ids, attention_mask):
        x = self.embedding(input_ids)
        x = x + self.position_embedding[:, :x.size(1), :]
        src_key_padding_mask = ~attention_mask.bool()
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x.mean(dim=1)


class DrugAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DrugAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, finger_enc, physico_enc):
        x = torch.stack([finger_enc, physico_enc], dim=1)
        out, w = self.attn(x, x, x)
        return out.mean(dim=1), w.mean(dim=0)


class DrugIntegrationAutoencoderWithAttention(nn.Module):
    def __init__(self, finger_input_dim, physico_input_dim, finger_encoding_dim=256, physico_encoding_dim=256, latent_encoding_dim=512):
        super(DrugIntegrationAutoencoderWithAttention, self).__init__()
        self.encoder1 = make_mlp(finger_input_dim, finger_encoding_dim, hidden_dims=[1024])
        self.encoder2 = make_mlp(physico_input_dim, physico_encoding_dim, hidden_dims=[256])
        self.proj1 = nn.Linear(finger_encoding_dim, latent_encoding_dim)
        self.proj2 = nn.Linear(physico_encoding_dim, latent_encoding_dim)
        self.attention_fusion = DrugAttentionFusion(embed_dim=latent_encoding_dim, num_heads=4)
        self.decoder1 = make_mlp(latent_encoding_dim, finger_input_dim, hidden_dims=[1024])
        self.decoder2 = make_mlp(latent_encoding_dim, physico_input_dim, hidden_dims=[256])

    def forward(self, finger_input, physico_input):
        finger_encoded = self.encoder1(finger_input)
        physico_encoded = self.encoder2(physico_input)
        finger_aligned = self.proj1(finger_encoded)
        physico_aligned = self.proj2(physico_encoded)
        fused, attn = self.attention_fusion(finger_aligned, physico_aligned)
        return self.decoder1(fused), self.decoder2(fused), attn


class AttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, rna, sga, protein):
        stacked = torch.stack([rna, sga, protein], dim=1)
        Q, K, V = self.query(stacked), self.key(stacked), self.value(stacked)
        score = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        w = self.softmax(score)
        return torch.matmul(w, V).mean(dim=1), w.mean(dim=0)


class IntegrationAutoencoderWithAttention(nn.Module):
    def __init__(self, rna_input_dim, sga_input_dim, protein_input_dim, rna_encoding_dim=512, sga_encoding_dim=512, protein_encoding_dim=512, latent_encoding_dim=512):
        super(IntegrationAutoencoderWithAttention, self).__init__()
        self.encoder1 = make_mlp(rna_input_dim, rna_encoding_dim, hidden_dims=[1024])
        self.encoder2 = make_mlp(sga_input_dim, sga_encoding_dim, hidden_dims=[1024])
        self.encoder3 = make_mlp(protein_input_dim, protein_encoding_dim, hidden_dims=[1024])
        self.proj_rna = nn.Linear(rna_encoding_dim, latent_encoding_dim)
        self.proj_sga = nn.Linear(sga_encoding_dim, latent_encoding_dim)
        self.proj_protein = nn.Linear(protein_encoding_dim, latent_encoding_dim)
        self.attention_fusion = AttentionFusion(input_dim=latent_encoding_dim, hidden_dim=latent_encoding_dim)
        self.decoder1 = make_mlp(latent_encoding_dim, rna_input_dim, hidden_dims=[1024])
        self.decoder2 = make_mlp(latent_encoding_dim, sga_input_dim, hidden_dims=[1024])
        self.decoder3 = make_mlp(latent_encoding_dim, protein_input_dim, hidden_dims=[1024])

    def forward(self, rna_input, sga_input, protein_input):
        rna_encoded = self.encoder1(rna_input)
        sga_encoded = self.encoder2(sga_input)
        protein_encoded = self.encoder3(protein_input)
        rna_aligned = self.proj_rna(rna_encoded)
        sga_aligned = self.proj_sga(sga_encoded)
        protein_aligned = self.proj_protein(protein_encoded)
        fused, attn = self.attention_fusion(rna_aligned, sga_aligned, protein_aligned)
        return self.decoder1(fused), self.decoder2(fused), self.decoder3(fused), attn


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = make_mlp(input_dim, 1, hidden_dims=[256, 128, 64, 32, 16], dropout=0.3)

    def forward(self, x):
        return self.model(x)


class StudentFusionDrugResponseModel(nn.Module):
    def __init__(self, vae, drug_ae, smiles_encoder, omics_ae, regressor):
        super(StudentFusionDrugResponseModel, self).__init__()
        self.vae = vae
        self.drug_ae = drug_ae
        self.smiles_encoder = smiles_encoder
        self.omics_ae = omics_ae
        self.regressor = regressor

    def forward(self, finger_input, physico_input, smiles_input_ids, smiles_attention_mask, rna_input):
        _, sga_gen, protein_gen, _, _ = self.vae(rna_input)
        drug_fused, drug_w = self.drug_ae.attention_fusion(
            self.drug_ae.proj1(self.drug_ae.encoder1(finger_input)),
            self.drug_ae.proj2(self.drug_ae.encoder2(physico_input))
        )
        smiles_embedding = self.smiles_encoder.encode_embedding(smiles_input_ids, smiles_attention_mask)
        drug_all = torch.cat([drug_fused, smiles_embedding], dim=1)
        omics_fused, omics_w = self.omics_ae.attention_fusion(
            self.omics_ae.proj_rna(self.omics_ae.encoder1(rna_input)),
            self.omics_ae.proj_sga(self.omics_ae.encoder2(sga_gen)),
            self.omics_ae.proj_protein(self.omics_ae.encoder3(protein_gen))
        )
        final_input = torch.cat([drug_all, omics_fused], dim=1)
        pred_ic50 = self.regressor(final_input)
        return pred_ic50, {'omics_w': omics_w, 'drug_w': drug_w}

    def freeze_backbone(self):
        for param in self.drug_ae.parameters():
            param.requires_grad = False
        for param in self.omics_ae.parameters():
            param.requires_grad = False
        for param in self.smiles_encoder.parameters():
            param.requires_grad = False

    def train_regressor(self, mode=True):
        super().train(mode)
        self.drug_ae.eval()
        self.omics_ae.eval()
        self.smiles_encoder.eval()
        return self

    def eval_all(self):
        super().eval()
        self.vae.eval()
        self.drug_ae.eval()
        self.omics_ae.eval()
        self.smiles_encoder.eval()
        return self


class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list, label_tensor, softlabels_tensor):
        self.samples = sample_list
        self.labels = label_tensor
        self.softlabels = softlabels_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["finger"],
            s["physico"],
            s["smiles_id"],
            s["smiles_mask"],
            s["rna"],
            self.labels[idx],
            self.softlabels[idx]
        )


def mask_input_ids(input_ids, mask_token_id, mask_prob=0.15):
    input_ids_masked = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask = (rand < mask_prob) & (input_ids != 0)
    input_ids_masked[mask] = mask_token_id
    return input_ids_masked, mask


def compute_distillation_loss(pred, y_true, y_soft, alpha=0.5):
    mse = nn.MSELoss()
    loss_hard = mse(pred, y_true)
    loss_soft = mse(pred, y_soft)
    return alpha * loss_soft + (1 - alpha) * loss_hard


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

rna_encoding_dim = 512
sga_encoding_dim = 512
protein_encoding_dim = 512
finger_encoding_dim = 512
physico_encoding_dim = 512
drug_latent_dim = 512
omics_latent_dim = 512
d_model = 128

regress_epochs = 300
batch_size = 512
regress_lr = 1e-3
alpha = 0.5

# Load data
df_cell_rna = pd.read_csv("../data/source/gdsc_rna_filter_data.csv")
df_soft = pd.read_csv("../data/result/teacher_model_predictions.csv")
df_drug_physico_features = pd.read_csv("../data/source/gdsc_advanced_physicochemical_descriptors.csv")
df_smiles = pd.read_csv("../data/source/filter_gdsc_cid_smiles.csv")

# Generate ECFP fingerprints from SMILES
print("Generating ECFP fingerprints from SMILES...")
df_drug_finger_features = generate_fingerprint_features(df_smiles)
print(f"Generated ECFP fingerprints for {len(df_drug_finger_features)} drugs")

# Standardize features
scaler = StandardScaler()
X_cell_rna = scaler.fit_transform(df_cell_rna.iloc[:, 1:])
X_drug_finger = scaler.fit_transform(df_drug_finger_features.iloc[:, 1:])
X_drug_physico = scaler.fit_transform(df_drug_physico_features.iloc[:, 1:])

X_cell_rna_tensor = torch.tensor(X_cell_rna, dtype=torch.float32).to(device)
X_drug_finger_tensor = torch.tensor(X_drug_finger, dtype=torch.float32).to(device)
X_drug_physico_tensor = torch.tensor(X_drug_physico, dtype=torch.float32).to(device)

# Load pre-trained models
vae = VAE(
    input_dim=X_cell_rna.shape[1],
    sga_dim=X_cell_rna.shape[1],
    protein_dim=X_cell_rna.shape[1],
    latent_dim=64,
    hidden_dim=256
).to(device)
vae.load_state_dict(torch.load("../model/vae_model.pth"))

omics_model = IntegrationAutoencoderWithAttention(
    rna_input_dim=X_cell_rna.shape[1],
    sga_input_dim=X_cell_rna.shape[1],
    protein_input_dim=X_cell_rna.shape[1],
    rna_encoding_dim=rna_encoding_dim,
    sga_encoding_dim=sga_encoding_dim,
    protein_encoding_dim=protein_encoding_dim,
    latent_encoding_dim=omics_latent_dim
).to(device)
omics_model.load_state_dict(torch.load("../model/teacher_omics_ae.pth"))

drug_model = DrugIntegrationAutoencoderWithAttention(
    finger_input_dim=X_drug_finger.shape[1],  # Now 2048 for ECFP
    physico_input_dim=X_drug_physico.shape[1],
    finger_encoding_dim=finger_encoding_dim,
    physico_encoding_dim=physico_encoding_dim,
    latent_encoding_dim=drug_latent_dim
).to(device)
drug_model.load_state_dict(torch.load("../model/teacher_drug_ae.pth"))

regressor_model = RegressionModel(input_dim=omics_latent_dim + drug_latent_dim + d_model).to(device)
tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
smiles_model = DrugEncoder(vocab_size=tokenizer.vocab_size, d_model=d_model).to(device)
smiles_model.load_state_dict(torch.load("../model/teacher_smiles_encoder.pth"))

# Create dictionaries for mapping
cell_rna_dict = dict(zip(df_cell_rna['cell_line_name'], range(len(df_cell_rna))))
drug_finger_dict = dict(zip(df_drug_finger_features['pubchem_cid'], range(len(df_drug_finger_features))))
drug_physico_dict = dict(zip(df_drug_physico_features['pubchem_cid'], range(len(df_drug_physico_features))))
smiles_dict = dict(zip(df_smiles['pubchem_cid'], df_smiles['smiles']))

# Prepare tensors for dataset creation
X_cell_rna_tensor_pre = torch.tensor(X_cell_rna, dtype=torch.float32)
X_drug_finger_tensor_pre = torch.tensor(X_drug_finger, dtype=torch.float32)
X_drug_physico_tensor_pre = torch.tensor(X_drug_physico, dtype=torch.float32)

# Create dataset samples
encoded_cache = {}
samples = []
labels = []
softlabels = []

valid_rows = []
for _, row in df_soft.iterrows():
    cid = row['pubchem_cid']
    cname = row['cell_line_name']
    if (cname in cell_rna_dict and cid in drug_finger_dict and cid in smiles_dict):
        valid_rows.append((row, row['y_transformed'], row['predict']))

print(f"Found {len(valid_rows)} valid drug-cell line pairs")

for row_data in valid_rows:
    row, y_transformed, y_predict = row_data
    cid = row['pubchem_cid']
    cname = row['cell_line_name']
    c_idx = cell_rna_dict[cname]
    d_idx = drug_finger_dict[cid]
    smiles = smiles_dict[cid]
    
    if smiles not in encoded_cache:
        encoded = tokenizer(smiles, padding='max_length', truncation=True, max_length=114, return_tensors='pt')
        encoded_cache[smiles] = (encoded['input_ids'][0], encoded['attention_mask'][0])
    
    smiles_id, smiles_mask = encoded_cache[smiles]
    
    sample = {
        "finger": X_drug_finger_tensor_pre[d_idx],
        "physico": X_drug_physico_tensor_pre[d_idx],
        "smiles_id": smiles_id,
        "smiles_mask": smiles_mask,
        "rna": X_cell_rna_tensor_pre[c_idx]
    }
    samples.append(sample)
    labels.append(y_transformed)
    softlabels.append(y_predict)

# Initialize student model
fusion_regressor_model = StudentFusionDrugResponseModel(
    vae=vae,
    drug_ae=drug_model,
    smiles_encoder=smiles_model,
    omics_ae=omics_model,
    regressor=regressor_model
).to(device)

# Split data
X_train_val_raw, X_test_raw, y_train_val_raw, y_test_raw, y_train_val_soft, y_test_soft = train_test_split(
    samples, labels, softlabels, test_size=0.1, random_state=seed
)
X_train_raw, X_val_raw, y_train_raw, y_val_raw, y_train_soft, y_val_soft = train_test_split(
    X_train_val_raw, y_train_val_raw, y_train_val_soft, test_size=1/9, random_state=seed
)

y_train_label = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
y_val_label = torch.tensor(y_val_raw, dtype=torch.float32).unsqueeze(1)
y_test_label = torch.tensor(y_test_raw, dtype=torch.float32).unsqueeze(1)

y_train_slabel = torch.tensor(np.array(y_train_soft).reshape(-1, 1), dtype=torch.float32)
y_val_slabel = torch.tensor(np.array(y_val_soft).reshape(-1, 1), dtype=torch.float32)
y_test_slabel = torch.tensor(np.array(y_test_soft).reshape(-1, 1), dtype=torch.float32)

# Create datasets and dataloaders
train_dataset = FusionDataset(X_train_raw, y_train_label, y_train_slabel)
val_dataset = FusionDataset(X_val_raw, y_val_label, y_val_slabel)
test_dataset = FusionDataset(X_test_raw, y_test_label, y_test_slabel)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train regression model
fusion_regressor_model.freeze_backbone()
optimizer = torch.optim.Adam(fusion_regressor_model.parameters(), lr=regress_lr)

patience = 20
best_val_loss = float('inf')
counter = 0

print("Training student regression model...")
for epoch in range(regress_epochs):
    total_loss = 0
    fusion_regressor_model.train_regressor()
    
    for finger, physico, smiles_id, smiles_mask, rna, label, slabel in train_loader:
        finger = finger.to(device)
        physico = physico.to(device)
        smiles_id = smiles_id.to(device)
        smiles_mask = smiles_mask.to(device)
        rna = rna.to(device)
        label = label.to(device)
        slabel = slabel.to(device)
        
        pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna)
        loss = compute_distillation_loss(pred, label, slabel, alpha)
        
        assert not torch.isnan(loss).any(), "NaN detected in loss"
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion_regressor_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    fusion_regressor_model.eval_all()
    val_loss = 0
    
    with torch.no_grad():
        for finger, physico, smiles_id, smiles_mask, rna, label, slabel in val_loader:
            finger = finger.to(device)
            physico = physico.to(device)
            smiles_id = smiles_id.to(device)
            smiles_mask = smiles_mask.to(device)
            rna = rna.to(device)
            label = label.to(device)
            
            pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna)
            val_loss += nn.MSELoss()(pred, label).item()
    
    val_loss /= len(val_loader)
    
    if (epoch + 1) % 10 == 0:
        print(f"[StudentFusionRegressor] Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save student model
torch.save(fusion_regressor_model.state_dict(), "../model/student_fusion_regressor_model.pth")

# Evaluate on test set
fusion_regressor_model.eval_all()

all_preds = []
all_labels = []

with torch.no_grad():
    for finger, physico, smiles_id, smiles_mask, rna, label, _ in test_loader:
        finger = finger.to(device)
        physico = physico.to(device)
        smiles_id = smiles_id.to(device)
        smiles_mask = smiles_mask.to(device)
        rna = rna.to(device)
        label = label.to(device)
        
        pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna)
        all_preds.append(pred.cpu())
        all_labels.append(label.cpu())

all_preds = torch.cat(all_preds, dim=0).squeeze(1).numpy()
all_labels = torch.cat(all_labels, dim=0).squeeze(1).numpy()



# Load scalers and convert to original scale
import joblib

scaler_y = joblib.load("../model/scaler_y.pkl")
qt_y = joblib.load("../model/qt_y.pkl")

all_preds_original = scaler_y.inverse_transform(qt_y.inverse_transform(all_preds.reshape(-1, 1))).flatten()
all_labels_original = scaler_y.inverse_transform(qt_y.inverse_transform(all_labels.reshape(-1, 1))).flatten()

# Calculate metrics on original scale
mae_original = mean_absolute_error(all_labels_original, all_preds_original)
mse_original = mean_squared_error(all_labels_original, all_preds_original)
rmse_original = np.sqrt(mse_original)
r2_original = r2_score(all_labels_original, all_preds_original)
pearson_original = pearsonr(all_labels_original, all_preds_original)[0]
spearman_original = spearmanr(all_labels_original, all_preds_original)[0]

print("\n=== Evaluation Results (Original Scale) ===")
print(f"MAE: {mae_original}")
print(f"MSE: {mse_original}")
print(f"RMSE: {rmse_original}")
print(f"R2: {r2_original}")
print(f"Pearson: {pearson_original}")
print(f"Spearman: {spearman_original}")

# Save evaluation results
eval_results = pd.DataFrame({
    'metric': ['MAE', 'MSE', 'RMSE', 'R2', 'Pearson', 'Spearman'],
    'original': [mae_original, mse_original, rmse_original, r2_original, pearson_original, spearman_original]
})

eval_path = "../model/student_evaluation_results.csv"
eval_results.to_csv(eval_path, index=False)
