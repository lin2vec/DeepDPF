import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

script_name = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join('output', script_name)
log_dir = output_dir
os.makedirs(output_dir, exist_ok=True)

log_filename = f"{script_name}_training.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger()
logger.info(f"Running script: {script_name}")


class PairwiseDrugDataset(Dataset):
    def __init__(self, drug_feat_path, label_csv_path):
        self.drug_feats = np.load(drug_feat_path)
        scaler_drug = StandardScaler()
        self.drug_feats = scaler_drug.fit_transform(self.drug_feats)
        self.drug_feats = torch.tensor(self.drug_feats, dtype=torch.float)

        df = pd.read_csv(label_csv_path)
        labels_matrix = df.iloc[:, 1:].values

        valid_mask = ~np.isnan(labels_matrix)

        prot_indices, drug_indices = np.where(valid_mask)
        valid_values = labels_matrix[valid_mask]

        self.prot_indices = torch.tensor(prot_indices, dtype=torch.long)
        self.drug_indices = torch.tensor(drug_indices, dtype=torch.long)
        self.labels = torch.tensor(valid_values, dtype=torch.float)

        logger.info(f"Loaded {len(self.labels)} valid samples.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        p_idx = self.prot_indices[idx]
        d_idx = self.drug_indices[idx]

        drug_feat = self.drug_feats[d_idx]
        label = self.labels[idx]

        return p_idx, drug_feat, label


class RelationGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations, heads=4):
        super().__init__()
        head_dim = out_channels // heads

        self.convs = nn.ModuleList([
            GATConv(in_channels, head_dim, heads=heads, concat=True)
            for _ in range(num_relations)
        ])

    def forward(self, x, edge_index, edge_type):
        out = 0
        for i, conv in enumerate(self.convs):
            mask = edge_type == i
            edge_index_i = edge_index[:, mask]

            if edge_index_i.size(1) > 0:
                out += conv(x, edge_index_i)
        return out


class LayerAttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_layer = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states):
        scores = self.attention_layer(hidden_states)
        weights = F.softmax(scores, dim=0)
        fused_features = torch.sum(weights * hidden_states, dim=0)
        return fused_features


class PairwiseResponseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = RelationGATConv(in_channels=1024, out_channels=256, num_relations=2, heads=4)
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv2 = RelationGATConv(in_channels=256, out_channels=256, num_relations=2, heads=4)
        self.norm2 = nn.LayerNorm(256)
        self.dropout2 = nn.Dropout(p=0.2)

        self.conv3 = RelationGATConv(in_channels=256, out_channels=256, num_relations=2, heads=4)
        self.norm3 = nn.LayerNorm(256)
        self.dropout3 = nn.Dropout(p=0.2)

        self.skip_proj = nn.Linear(1024, 256)
        self.drug_proj = nn.Linear(768, 256)

        self.layer_fusion = LayerAttentionFusion(hidden_dim=256)

        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(256)

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.mlp_feature = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(p=0.3)
        )

        self.predictor = nn.Linear(128, 1)

    def forward(self, graph_data, prot_idx_batch, drug_feat_batch):
        x = graph_data.x
        edge_index = graph_data.edge_index
        edge_type = graph_data.edge_type

        h0 = self.skip_proj(x)
        h0 = F.relu(h0)

        h1 = self.conv1(x, edge_index, edge_type)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout1(h1)

        h2 = self.conv2(h1, edge_index, edge_type)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout2(h2)

        h3 = self.conv3(h2, edge_index, edge_type)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)
        h3 = self.dropout3(h3)

        all_layers = torch.stack([h0, h1, h2, h3], dim=0)
        prot_embed_full = self.layer_fusion(all_layers)
        prot_embed_batch = prot_embed_full[prot_idx_batch]

        drug_embed_batch = self.drug_proj(drug_feat_batch)
        drug_embed_batch = F.relu(drug_embed_batch)

        prot_seq = prot_embed_batch.unsqueeze(1)
        drug_seq = drug_embed_batch.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=prot_seq, key=drug_seq, value=drug_seq)
        prot_attended = self.attn_norm(prot_seq + attn_out)

        combined_seq = torch.cat([prot_attended, drug_seq], dim=1)
        trans_out = self.transformer(combined_seq)

        trans_out_flat = trans_out.view(trans_out.size(0), -1)

        final_feat = torch.cat([prot_embed_batch, drug_embed_batch, trans_out_flat], dim=-1)

        hidden_features = self.mlp_feature(final_feat)

        out = self.predictor(hidden_features)

        return out.squeeze(-1)


FC_THRESHOLD = np.log2(1.5)


def concordance_index(prediction, target):
    pred_np = prediction.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    valid_mask = ~(np.isnan(pred_np) | np.isnan(target_np))
    pred_np = pred_np[valid_mask]
    target_np = target_np[valid_mask]

    n = len(target_np)
    if n < 2:
        return 0.0

    order = np.argsort(target_np, kind='mergesort')
    target_sorted = target_np[order]
    pred_sorted = pred_np[order]

    unique_preds = np.unique(pred_sorted)
    pred_ranks = np.searchsorted(unique_preds, pred_sorted) + 1
    tree = np.zeros(len(unique_preds) + 1, dtype=np.int64)

    def tree_add(index, value):
        while index < len(tree):
            tree[index] += value
            index += index & -index

    def tree_sum(index):
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    concordant = 0.0
    comparable = 0
    previous_count = 0
    i = 0

    while i < n:
        j = i + 1
        while j < n and target_sorted[j] == target_sorted[i]:
            j += 1

        for rank in pred_ranks[i:j]:
            less_count = tree_sum(rank - 1)
            equal_count = tree_sum(rank) - less_count
            concordant += less_count + 0.5 * equal_count
            comparable += previous_count

        for rank in pred_ranks[i:j]:
            tree_add(rank, 1)

        previous_count += j - i
        i = j

    if comparable == 0:
        return 0.0

    return concordant / comparable


def evaluate_metrics(prediction, target):
    if len(target) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    mse = F.mse_loss(prediction, target).item()
    mae = F.l1_loss(prediction, target).item()

    pred_mean = torch.mean(prediction)
    target_mean = torch.mean(target)

    num = torch.sum((prediction - pred_mean) * (target - target_mean))
    den = torch.sqrt(torch.sum((prediction - pred_mean) ** 2) * torch.sum((target - target_mean) ** 2))
    pearson = (num / (den + 1e-8)).item()

    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    r2 = (1 - ss_res / (ss_tot + 1e-8)).item()
    ci = concordance_index(prediction, target)

    return pearson, r2, ci, mse, mae


def hybrid_loss(prediction, target):
    mask = (torch.abs(target) >= FC_THRESHOLD).float()
    weight = mask * 4.0 + 1.0
    weighted_mse = torch.mean(weight * (prediction - target) ** 2)

    pred_mean = torch.mean(prediction)
    target_mean = torch.mean(target)
    num = torch.sum((prediction - pred_mean) * (target - target_mean))
    den = torch.sqrt(torch.sum((prediction - pred_mean) ** 2) * torch.sum((target - target_mean) ** 2))
    pearson_val = num / (den + 1e-8)
    pearson_loss = 1.0 - pearson_val

    total_loss = weighted_mse + 0.5 * pearson_loss
    return total_loss


if __name__ == '__main__':
    drug_feature_file = 'database/drug_feature.npy'
    csv_label_file = 'database/drug_protein_log2FC.csv'
    graph_file = 'database/string_ppi_graph.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    full_dataset = PairwiseDrugDataset(drug_feature_file, csv_label_file)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size],
                                                            generator=generator)

    BATCH_SIZE = 8192
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    ppi_graph = torch.load(graph_file, weights_only=False)

    scaler_prot = StandardScaler()
    prot_feats_scaled = scaler_prot.fit_transform(ppi_graph.x.numpy())
    ppi_graph.x = torch.tensor(prot_feats_scaled, dtype=torch.float)
    ppi_graph = ppi_graph.to(device)

    model = PairwiseResponseModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_corr = -1.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    patience = 20
    no_improve_count = 0

    logger.info("Training started.")
    for epoch in range(300):
        model.train()

        for prot_idx, drug_feat, label in train_loader:
            optimizer.zero_grad()

            prot_idx = prot_idx.to(device)
            drug_feat = drug_feat.to(device)
            label = label.to(device)

            prediction = model(ppi_graph, prot_idx, drug_feat)

            loss = hybrid_loss(prediction, label)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for prot_idx, drug_feat, label in val_loader:
                prot_idx = prot_idx.to(device)
                drug_feat = drug_feat.to(device)
                label = label.to(device)

                prediction = model(ppi_graph, prot_idx, drug_feat)
                all_preds.append(prediction)
                all_labels.append(label)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        val_corr, val_r2, val_ci, val_mse, val_mae = evaluate_metrics(all_preds, all_labels)

        mask = torch.abs(all_labels) >= FC_THRESHOLD
        if torch.sum(mask) > 0:
            val_sig_corr, val_sig_r2, val_sig_ci, val_sig_mse, val_sig_mae = evaluate_metrics(all_preds[mask], all_labels[mask])
        else:
            val_sig_corr, val_sig_r2, val_sig_ci, val_sig_mse, val_sig_mae = 0.0, 0.0, 0.0, 0.0, 0.0

        logger.info(f"Epoch {epoch + 1} completed.")
        logger.info(f"Validation (all) -> Pearson: {val_corr:.4f} | R2: {val_r2:.4f} | CI: {val_ci:.4f} | MSE: {val_mse:.4f} | MAE: {val_mae:.4f}")
        logger.info(
            f"Validation (significant) -> Pearson: {val_sig_corr:.4f} | R2: {val_sig_r2:.4f} | CI: {val_sig_ci:.4f} | MSE: {val_sig_mse:.4f} | MAE: {val_sig_mae:.4f}")

        scheduler.step(val_sig_corr)

        if val_sig_corr > best_val_corr:
            best_val_corr = val_sig_corr
            torch.save(model.state_dict(), best_model_path)
            logger.info("Best model saved.")
            no_improve_count = 0
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epoch(s).")
            if no_improve_count >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement.")
                break

    logger.info("Final testing started.")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for prot_idx, drug_feat, label in test_loader:
            prot_idx = prot_idx.to(device)
            drug_feat = drug_feat.to(device)
            label = label.to(device)

            prediction = model(ppi_graph, prot_idx, drug_feat)

            test_preds.append(prediction)
            test_labels.append(label)

    test_preds = torch.cat(test_preds)
    test_labels = torch.cat(test_labels)

    test_corr, test_r2, test_ci, test_mse, test_mae = evaluate_metrics(test_preds, test_labels)

    mask = torch.abs(test_labels) >= FC_THRESHOLD
    if torch.sum(mask) > 0:
        test_sig_corr, test_sig_r2, test_sig_ci, test_sig_mse, test_sig_mae = evaluate_metrics(test_preds[mask], test_labels[mask])
    else:
        test_sig_corr, test_sig_r2, test_sig_ci, test_sig_mse, test_sig_mae = 0.0, 0.0, 0.0, 0.0, 0.0

    logger.info("Final test results:")
    logger.info(f"Test (all) -> Pearson: {test_corr:.4f} | R2: {test_r2:.4f} | CI: {test_ci:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}")
    logger.info(f"Test (significant) -> Pearson: {test_sig_corr:.4f} | R2: {test_sig_r2:.4f} | CI: {test_sig_ci:.4f} | MSE: {test_sig_mse:.4f} | MAE: {test_sig_mae:.4f}")
