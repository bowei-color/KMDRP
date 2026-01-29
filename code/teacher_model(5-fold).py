# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import random
from transformers import RobertaTokenizerFast
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem




# MLP creation helper function
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

# Drug Encoder
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
        # input_ids: [B, L], attention_mask: [B, L]
        x = self.embedding(input_ids)                             # [B, L, D]
        x = x + self.position_embedding[:, :x.size(1), :]         # [B, L, D] + position_embedding
        src_key_padding_mask = ~attention_mask.bool()             # [B, L]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.output_layer(x)

    def encode_embedding(self, input_ids, attention_mask):
        # Same forward pass, but returns embedding
        x = self.embedding(input_ids)
        x = x + self.position_embedding[:, :x.size(1), :]
        src_key_padding_mask = ~attention_mask.bool()
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x.mean(dim=1)

# Drug Attention Fusion
class DrugAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DrugAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, finger_enc, physico_enc):
        x = torch.stack([finger_enc, physico_enc], dim=1)
        out, w = self.attn(x, x, x)
        return out.mean(dim=1), w.mean(dim=0)

# Drug Integration Autoencoder with Attention
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

# Attention Fusion for Omics
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


# Integration Autoencoder with Attention (Omics)
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


# Regression Model (Fully connected network)
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = make_mlp(input_dim, 1, hidden_dims=[256, 128, 64, 32, 16], dropout=0.3)

    def forward(self, x):
        return self.model(x)
    
    
class TeacherFusionDrugResponseModel(nn.Module):
    def __init__(self, drug_ae, smiles_encoder, omics_ae, regressor):
        super(TeacherFusionDrugResponseModel, self).__init__()
        self.drug_ae = drug_ae
        self.smiles_encoder = smiles_encoder
        self.omics_ae = omics_ae
        self.regressor = regressor

    def forward(self, finger_input, physico_input, smiles_input_ids, smiles_attention_mask,
                rna_input, sga_input, protein_input):
        drug_fused, drug_w = self.drug_ae.attention_fusion(
            self.drug_ae.proj1(self.drug_ae.encoder1(finger_input)),
            self.drug_ae.proj2(self.drug_ae.encoder2(physico_input))
        )
        smiles_embedding = self.smiles_encoder.encode_embedding(smiles_input_ids, smiles_attention_mask)
        drug_all = torch.cat([drug_fused, smiles_embedding], dim=1)
        omics_fused, omics_w = self.omics_ae.attention_fusion(
            self.omics_ae.proj_rna(self.omics_ae.encoder1(rna_input)),
            self.omics_ae.proj_sga(self.omics_ae.encoder2(sga_input)),
            self.omics_ae.proj_protein(self.omics_ae.encoder3(protein_input))
        )
        final_input = torch.cat([drug_all, omics_fused], dim=1)
        pred_ic50 = self.regressor(final_input)
        return pred_ic50, {'omics_w': omics_w, 'drug_w': drug_w}

    def freeze_backbone(self):
        for p in [self.drug_ae, self.omics_ae, self.smiles_encoder]:
            for param in p.parameters():
                param.requires_grad = False

    def train_regressor(self, mode=True):
        super().train(mode)
        self.drug_ae.eval()
        self.omics_ae.eval()
        self.smiles_encoder.eval()
        return self

    def eval_all(self):
        super().eval()
        self.drug_ae.eval()
        self.omics_ae.eval()
        self.smiles_encoder.eval()
        return self
    

class FusionDataset(torch.utils.data.Dataset):
    def __init__(self, sample_list, label_tensor):
        self.samples = sample_list
        self.labels = label_tensor

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
            s["sga"],
            s["protein"],
            self.labels[idx]
        )
    
# Masking Input Function (for SMILES)
def mask_input_ids(input_ids, mask_token_id, mask_prob=0.15):
    """Randomly mask input tokens for self-supervised learning"""
    input_ids_masked = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask = (rand < mask_prob) & (input_ids != 0)  # avoid padding
    input_ids_masked[mask] = mask_token_id
    return input_ids_masked, mask


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


seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters and dataset paths
rna_encoding_dim = 512
sga_encoding_dim = 512
protein_encoding_dim = 512
finger_encoding_dim = 512
physico_encoding_dim = 512
drug_latent_dim = 512
omics_latent_dim = 512
d_model = 128

drug_epochs = 2000
omics_epochs = 2000
smiles_epochs = 200
regress_epochs = 200
batch_size = 512

omics_lr = 1e-3
drug_lr = 1e-3
smiles_lr = 1e-3
regress_lr = 1e-3

# Load data
df_cell_rna = pd.read_csv("../data/source/gdsc_rna_filter_data.csv")
df_cell_sga = pd.read_csv("../data/source/gdsc_sga_filter_data.csv")
df_cell_protein = pd.read_csv("../data/source/gdsc_protein_filter_data.csv")
df_ic50 = pd.read_csv("../data/source/ic50_filter_data.csv")
df_drug_physico_features = pd.read_csv("../data/source/gdsc_advanced_physicochemical_descriptors.csv")
df_smiles = pd.read_csv("../data/source/filter_gdsc_cid_smiles.csv")

# Generate ECFP fingerprints from SMILES
df_drug_finger_features = generate_fingerprint_features(df_smiles)

scaler = StandardScaler()
X_cell_rna = scaler.fit_transform(df_cell_rna.iloc[:, 1:])
X_cell_sga = scaler.fit_transform(df_cell_sga.iloc[:, 1:])
X_cell_protein = scaler.fit_transform(df_cell_protein.iloc[:, 1:])
X_drug_finger = scaler.fit_transform(df_drug_finger_features.iloc[:, 1:])
X_drug_physico = scaler.fit_transform(df_drug_physico_features.iloc[:, 1:])


X_cell_rna_tensor = torch.tensor(X_cell_rna, dtype=torch.float32).to(device)
X_cell_sga_tensor = torch.tensor(X_cell_sga, dtype=torch.float32).to(device)
X_cell_protein_tensor = torch.tensor(X_cell_protein, dtype=torch.float32).to(device)
X_drug_finger_tensor = torch.tensor(X_drug_finger, dtype=torch.float32).to(device)
X_drug_physico_tensor = torch.tensor(X_drug_physico, dtype=torch.float32).to(device)


omics_model = IntegrationAutoencoderWithAttention(
    rna_input_dim=X_cell_rna.shape[1],
    sga_input_dim=X_cell_sga.shape[1],
    protein_input_dim=X_cell_protein.shape[1],
    rna_encoding_dim = rna_encoding_dim,
    sga_encoding_dim = sga_encoding_dim,
    protein_encoding_dim = protein_encoding_dim,
    latent_encoding_dim = omics_latent_dim
).to(device)

drug_model = DrugIntegrationAutoencoderWithAttention(
    finger_input_dim=X_drug_finger.shape[1],  # Now 2048 for ECFP
    physico_input_dim=X_drug_physico.shape[1],
    finger_encoding_dim = finger_encoding_dim,
    physico_encoding_dim = physico_encoding_dim,
    latent_encoding_dim = drug_latent_dim
).to(device)

loss_fn = nn.MSELoss()
optimizer_omics = optim.Adam(omics_model.parameters(), lr=omics_lr)
optimizer_drug = optim.Adam(drug_model.parameters(), lr=drug_lr)


for epoch in range(omics_epochs):
    omics_model.train()
    optimizer_omics.zero_grad()
    out_rna, out_sga, out_protein, _ = omics_model(X_cell_rna_tensor, X_cell_sga_tensor, X_cell_protein_tensor)
    loss = loss_fn(out_rna, X_cell_rna_tensor) + loss_fn(out_sga, X_cell_sga_tensor) + loss_fn(out_protein, X_cell_protein_tensor)
    loss.backward()
    optimizer_omics.step()
    if (epoch + 1) % 100 == 0:
        print(f"[Omics AE] Epoch {epoch + 1}, Loss: {loss.item():.4f}")

for epoch in range(drug_epochs):
    drug_model.train()
    optimizer_drug.zero_grad()
    out_finger, out_physico, _ = drug_model(X_drug_finger_tensor, X_drug_physico_tensor)
    loss = loss_fn(out_finger, X_drug_finger_tensor) + loss_fn(out_physico, X_drug_physico_tensor)
    loss.backward()
    optimizer_drug.step()
    if (epoch + 1) % 100 == 0:
        print(f"[Drug AE] Epoch {epoch + 1}, Loss: {loss.item():.4f}")


tokenizer = RobertaTokenizerFast.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
mask_token_id = tokenizer.mask_token_id
pad_token_id = tokenizer.pad_token_id
smiles_list = df_smiles['smiles'].astype(str).tolist()
encoded = tokenizer(smiles_list, padding='max_length', truncation=True, max_length=114, return_tensors='pt')
input_ids = encoded['input_ids'].to(device)
attention_mask = encoded['attention_mask'].to(device)
dataset = TensorDataset(input_ids, attention_mask)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

smiles_model = DrugEncoder(vocab_size=tokenizer.vocab_size, d_model=d_model).to(device)
optimizer_smiles = optim.Adam(smiles_model.parameters(), lr=smiles_lr)
loss_fn_ce = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

for epoch in range(smiles_epochs):
    smiles_model.train()
    total_loss = 0
    for ids, mask in dataloader:
        ids, mask = ids.to(device), mask.to(device)
        ids_masked, mask_arr = mask_input_ids(ids, mask_token_id)
        out = smiles_model(ids_masked, attention_mask=mask)

        logits = out.view(-1, out.size(-1))              # [B*L, V]
        target = ids.view(-1)                            # [B*L]
        mask_flat = mask_arr.view(-1)                    # [B*L]
        loss = F.cross_entropy(logits[mask_flat], target[mask_flat])

        optimizer_smiles.zero_grad()
        loss.backward()
        optimizer_smiles.step()
        total_loss += loss.item()


    if (epoch + 1) % 10 == 0:
        print(f"[SMILES] Epoch {epoch + 1}, Loss: {loss.item():.4f}")




cell_rna_dict = dict(zip(df_cell_rna['cell_line_name'], range(len(df_cell_rna))))
cell_sga_dict = dict(zip(df_cell_sga['cell_line_name'], range(len(df_cell_sga))))
cell_protein_dict = dict(zip(df_cell_protein['cell_line_name'], range(len(df_cell_protein))))
drug_finger_dict = dict(zip(df_drug_finger_features['pubchem_cid'], range(len(df_drug_finger_features))))
drug_physico_dict = dict(zip(df_drug_physico_features['pubchem_cid'], range(len(df_drug_physico_features))))
smiles_dict = dict(zip(df_smiles['pubchem_cid'], df_smiles['smiles']))

X_cell_rna_tensor_pre = torch.tensor(X_cell_rna, dtype=torch.float32)
X_cell_sga_tensor_pre = torch.tensor(X_cell_sga, dtype=torch.float32)
X_cell_protein_tensor_pre = torch.tensor(X_cell_protein, dtype=torch.float32)
X_drug_finger_tensor_pre = torch.tensor(X_drug_finger, dtype=torch.float32)
X_drug_physico_tensor_pre = torch.tensor(X_drug_physico, dtype=torch.float32)

encoded_cache = {}

samples = []
labels = []

valid_rows = []
for _, row in df_ic50.iterrows():
    cid = row['pubchem_cid']
    cname = row['cell_line_name']
    if (cname in cell_rna_dict and 
        cid in drug_finger_dict and 
        cid in smiles_dict):
        valid_rows.append(row)

for row in valid_rows:
    cid = row['pubchem_cid']
    cname = row['cell_line_name']
    ic50 = row['ic50']
    
    c_idx = cell_rna_dict[cname]
    d_idx = drug_finger_dict[cid]
    smiles = smiles_dict[cid]
    
    if smiles not in encoded_cache:
        encoded = tokenizer(smiles, padding='max_length', truncation=True, 
                          max_length=114, return_tensors='pt')
        encoded_cache[smiles] = (encoded['input_ids'][0], encoded['attention_mask'][0])
    
    smiles_id, smiles_mask = encoded_cache[smiles]
    
    sample = {
        "finger": X_drug_finger_tensor_pre[d_idx],
        "physico": X_drug_physico_tensor_pre[d_idx],
        "smiles_id": smiles_id,
        "smiles_mask": smiles_mask,
        "rna": X_cell_rna_tensor_pre[c_idx],
        "sga": X_cell_sga_tensor_pre[cell_sga_dict[cname]],
        "protein": X_cell_protein_tensor_pre[cell_protein_dict[cname]]
    }
    samples.append(sample)
    labels.append(ic50)


kf = KFold(n_splits=5, shuffle=True, random_state=seed)

fold_transformed_results = []
fold_original_results = []

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(range(len(samples)))):
    print(f"第{fold_idx+1}折")
    

    X_test_raw = [samples[i] for i in test_idx]
    y_test_raw = [labels[i] for i in test_idx]
    
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=1/9,
        random_state=seed + fold_idx
    )
    
    X_train_raw = [samples[i] for i in train_idx]
    y_train_raw = [labels[i] for i in train_idx]
    X_val_raw = [samples[i] for i in val_idx]
    y_val_raw = [labels[i] for i in val_idx]
    
    scaler_y = StandardScaler()
    qt_y = QuantileTransformer(output_distribution='uniform', random_state=seed + fold_idx)
    
    y_train_transformed = scaler_y.fit_transform(np.array(y_train_raw).reshape(-1, 1))
    y_train_transformed = qt_y.fit_transform(y_train_transformed)
    
    y_val_transformed = qt_y.transform(scaler_y.transform(np.array(y_val_raw).reshape(-1, 1)))
    
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_transformed, dtype=torch.float32)
    
    train_dataset = FusionDataset(X_train_raw, y_train_tensor)
    val_dataset = FusionDataset(X_val_raw, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    regressor_model = RegressionModel(input_dim=omics_latent_dim + drug_latent_dim + d_model).to(device)
    
    fusion_regressor_model = TeacherFusionDrugResponseModel(
        drug_ae=drug_model,
        smiles_encoder=smiles_model,
        omics_ae=omics_model,
        regressor=regressor_model
    ).to(device)
    
    fusion_regressor_model.freeze_backbone()
    optimizer = torch.optim.Adam(fusion_regressor_model.parameters(), lr=regress_lr)
    loss_fn = torch.nn.MSELoss()
    fusion_regressor_model.train_regressor()
    
    patience = 20
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(regress_epochs):
        fusion_regressor_model.train_regressor()
        total_loss = 0
        for finger, physico, smiles_id, smiles_mask, rna, sga, protein, label in train_loader:
            finger = finger.to(device)
            physico = physico.to(device)
            smiles_id = smiles_id.to(device)
            smiles_mask = smiles_mask.to(device)
            rna = rna.to(device)
            sga = sga.to(device)
            protein = protein.to(device)
            label = label.to(device)
            
            pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna, sga, protein)
            loss = loss_fn(pred, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        fusion_regressor_model.eval_all()
        val_loss = 0
        with torch.no_grad():
            for finger, physico, smiles_id, smiles_mask, rna, sga, protein, label in val_loader:
                finger = finger.to(device)
                physico = physico.to(device)
                smiles_id = smiles_id.to(device)
                smiles_mask = smiles_mask.to(device)
                rna = rna.to(device)
                sga = sga.to(device)
                protein = protein.to(device)
                label = label.to(device)
                
                pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna, sga, protein)
                val_loss += loss_fn(pred, label).item()
        
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    fusion_regressor_model.eval_all()
    

    val_preds_transformed = []
    val_labels_transformed = []
    
    with torch.no_grad():
        for finger, physico, smiles_id, smiles_mask, rna, sga, protein, label in val_loader:
            finger = finger.to(device)
            physico = physico.to(device)
            smiles_id = smiles_id.to(device)
            smiles_mask = smiles_mask.to(device)
            rna = rna.to(device)
            sga = sga.to(device)
            protein = protein.to(device)
            label = label.to(device)
            
            pred, _ = fusion_regressor_model(finger, physico, smiles_id, smiles_mask, rna, sga, protein)
            val_preds_transformed.extend(pred.cpu().numpy().flatten().tolist())
            val_labels_transformed.extend(label.cpu().numpy().flatten().tolist())
    
    val_preds_original = scaler_y.inverse_transform(
        qt_y.inverse_transform(np.array(val_preds_transformed).reshape(-1, 1))
    ).flatten()
    
    val_labels_original = np.array(y_val_raw).flatten()

    
    mse_original = mean_squared_error(val_labels_original, val_preds_original)
    rmse_original = np.sqrt(mse_original)
    mae_original = mean_absolute_error(val_labels_original, val_preds_original)
    r2_original = r2_score(val_labels_original, val_preds_original)
    pearson_original = pearsonr(val_labels_original, val_preds_original)[0]
    spearman_original = spearmanr(val_labels_original, val_preds_original)[0]
    

    fold_original_results.append([mae_original, mse_original, rmse_original, r2_original, pearson_original, spearman_original])
    
    print(f"验证集原始值: MAE: {mae_original:.4f}, MSE: {mse_original:.4f}, RMSE: {rmse_original:.4f}, R2: {r2_original:.4f}, Pearson: {pearson_original:.4f}, Spearman: {spearman_original:.4f}")
    print("-" * 40)

fold_original_array = np.array(fold_original_results)
avg_original = np.mean(fold_original_array, axis=0)
std_original = np.std(fold_original_array, axis=0)

summary_data = {
    'metric': ['MAE', 'MSE', 'RMSE', 'R2', 'Pearson', 'Spearman'],
    'original_mean': avg_original,
    'original_std': std_original
}

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv("../data/result/teacher_5fold_summary.csv", index=False)

print("五折交叉验证平均结果:")
print(f"MAE: {avg_original[2]:.4f} ± {std_original[2]:.4f}")
print(f"MSE: {avg_original[0]:.4f} ± {std_original[0]:.4f}")
print(f"RMSE: {avg_original[1]:.4f} ± {std_original[1]:.4f}")
print(f"R²: {avg_original[3]:.4f} ± {std_original[3]:.4f}")
print(f"Pearson: {avg_original[4]:.4f} ± {std_original[4]:.4f}")
print(f"Spearman: {avg_original[5]:.4f} ± {std_original[5]:.4f}")