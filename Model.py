import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import joblib
import pickle

# Read the merged dataset
print("Loading merged dataset...")
df = pd.read_csv("DataSet/merged_dataset.csv")
print("Dataset shape:", df.shape)

# Clean column names and labels
df.columns = df.columns.str.strip()
df['Label'] = df['Label'].str.strip()

# Map labels to groups
mapping = {
    'BENIGN': 'BENIGN',
    'DDoS': 'DDoS',
    'PortScan': 'PortScan',
    'DoS GoldenEye': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'DoS slowloris': 'DoS'
}
df['Label_grp'] = df['Label'].map(mapping).fillna('Other')
print("New class counts:")
print(df['Label_grp'].value_counts())

# Plot class distribution before undersampling
counts = df['Label_grp'].value_counts()
plt.figure()
plt.bar(counts.index, counts.values)
plt.xticks(rotation=45)
plt.title("Merged Class Distribution (pre-undersample)")
plt.ylabel("Flows")
plt.tight_layout()
plt.savefig("class_distribution_before.png")
plt.show()

# Undersample to balance classes
max_samples = 50000
dfs = []
for cls, grp in df.groupby('Label_grp'):
    if len(grp) > max_samples:
        grp = grp.sample(max_samples, random_state=42)
    dfs.append(grp)
df_bal = pd.concat(dfs, ignore_index=True)
print("After undersampling:")
print(df_bal['Label_grp'].value_counts())

# Plot class distribution after undersampling
dist_after = df_bal['Label_grp'].value_counts()
plt.figure(figsize=(6,4))
plt.bar(dist_after.index, dist_after.values)
plt.xticks(rotation=45)
plt.title('Merged Class Distribution (post-undersample)')
plt.ylabel('Number of Flows')
plt.tight_layout()
plt.savefig("class_distribution_after.png")
plt.show()

# Compare before and after undersampling
pre = df['Label_grp'].value_counts()
post = dist_after
labels = pre.index

plt.figure(figsize=(8,4))
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, [pre[l] for l in labels], width, label='Pre')
plt.bar(x + width/2, [post.get(l,0) for l in labels], width, label='Post')
plt.xticks(x, labels, rotation=45)
plt.ylabel('Flows')
plt.title('Class Counts Before vs After Undersample')
plt.legend()
plt.tight_layout()
plt.savefig("class_distribution_comparison.png")
plt.show()

# Data preprocessing
print("Preprocessing data...")
df_bal.fillna(0, inplace=True)
df_bal.replace([np.inf, -np.inf], 0, inplace=True)

# Feature extraction
feature_cols = df_bal.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if 'label' not in c.lower()]
X = df_bal[feature_cols].values

# Label encoding
le = LabelEncoder()
y = le.fit_transform(df_bal['Label_grp'].values)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Creating graph structure...")
# Create k-NN graph
k = 10
A = kneighbors_graph(X_scaled, n_neighbors=k,
                    mode='connectivity', include_self=False).tocoo()
edge_index = torch.tensor([A.row, A.col], dtype=torch.long)

# Create PyTorch Geometric data object
data = Data(
    x=torch.tensor(X_scaled, dtype=torch.float),
    edge_index=edge_index,
    y=torch.tensor(y, dtype=torch.long)
)
print(f"Graph: {data.num_nodes} nodes, {data.num_edges//2} undirected edges")

# Train-test split
num_nodes = data.num_nodes
idx = np.arange(num_nodes)
train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y
)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)
train_mask[train_idx] = True
test_mask[test_idx] = True

data.train_mask = train_mask
data.test_mask = test_mask

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, hid_feats)
        self.lin = torch.nn.Linear(hid_feats, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.lin(x), dim=1)

# Initialize model and training components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = GCN(data.num_node_features, 64, len(le.classes_)).to(device)
data = data.to(device)

# Compute class weights for imbalanced data
counts = Counter(y)
total = sum(counts.values())
weights = torch.tensor([total/counts[i] for i in range(len(le.classes_))],
                       dtype=torch.float, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss(weight=weights)

# Training loop
print("Starting training...")
history = {'loss': [], 'train_acc': [], 'test_acc': []}
best_test, patience, stall = 0.0, 10, 0

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = out.argmax(dim=1)
        train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).float().mean().item()
        test_acc = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()

    history['loss'].append(loss.item())
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)

    print(f"Epoch {epoch:03d} | Loss {loss:.4f} | "
          f"TrainAcc {train_acc:.4f} | TestAcc {test_acc:.4f}")

    if test_acc > best_test:
        best_test, stall = test_acc, 0
    else:
        stall += 1
        if stall >= patience:
            print("Early stopping.")
            break

# Plot training history
plt.figure()
plt.plot(history['loss'])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig("training_loss.png")
plt.show()

plt.figure()
plt.plot(history['train_acc'], label='Train')
plt.plot(history['test_acc'], label='Test')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("training_accuracy.png")
plt.show()

# Final evaluation
model.eval()
with torch.no_grad():
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1).cpu().numpy()

y_true = data.y.cpu().numpy()[test_idx]
y_pred = pred[test_idx]

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
plt.figure()
disp.plot(ax=plt.gca())
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save the trained model and preprocessors
print("Saving model and preprocessors...")

# Save the model state dict
torch.save(model.state_dict(), 'Saved_Model/gcn_model.pth')

# Save the complete model (architecture + weights)
torch.save(model, 'Saved_Model/gcn_model_complete.pth')

# Save preprocessors
joblib.dump(scaler, 'Saved_Model/scaler.pkl')
joblib.dump(le, 'Saved_Model/label_encoder.pkl')

# Save feature columns and other metadata
model_metadata = {
    'feature_columns': feature_cols,
    'num_features': len(feature_cols),
    'num_classes': len(le.classes_),
    'class_names': list(le.classes_),
    'k_neighbors': k,
    'hidden_features': 64
}

with open('Saved_Model/model_metadata.pkl', 'wb') as f:
    pickle.dump(model_metadata, f)

print("Model training completed and saved!")
print(f"Best test accuracy: {best_test:.4f}")
print("\nSaved files:")
print("- gcn_model.pth (model weights)")
print("- gcn_model_complete.pth (complete model)")
print("- scaler.pkl (feature scaler)")
print("- label_encoder.pkl (label encoder)")
print("- model_metadata.pkl (model metadata)")