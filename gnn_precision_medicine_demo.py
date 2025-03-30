# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

################################################################################
# 1) CREATE DUMMY DATA: NODES, EDGES, AND (FUSED) NODE FEATURES
################################################################################

# Suppose we have 8 "patients" (nodes), each with:
#    - imaging features of dimension 4
#    - text features of dimension 3
#    - "genomic" features of dimension 2
# We can pretend these were produced independently (e.g., via CNN or NLP model).
# We then do "late fusion" by concatenating them into a single 4+3+2=9-D vector.

num_patients = 8
imaging_dim = 4
text_dim = 3
genomic_dim = 2
fused_dim = imaging_dim + text_dim + genomic_dim  # 9

# Create dummy fused features for each node:
torch.manual_seed(42)  # for reproducibility
imaging_feats = torch.randn(num_patients, imaging_dim)
text_feats    = torch.randn(num_patients, text_dim)
genomic_feats = torch.randn(num_patients, genomic_dim)

node_features = torch.cat([imaging_feats, text_feats, genomic_feats], dim=1)
# shape = [8, 9]

# Next, define edges. In a real scenario, edges might represent:
# - Shared biomarkers
# - Similar phenotypes
# - Known protein-protein interactions, etc.
# We'll randomly create some edges here, for demonstration.

edge_index = torch.tensor([
    [0, 1, 2, 2, 3, 4, 5, 6, 6],  # source
    [1, 0, 2, 3, 4, 6, 0, 5, 7]   # target
], dtype=torch.long)
# This means there's an edge 0 <-> 1, 2 <-> 2 (self-loop), 2 -> 3, 3->4, 4->6, etc.

# We can label each patient with a "disease severity" or "treatment response" class.
# E.g., 2-class classification: 0=non-responder, 1=responder
labels = torch.tensor([1, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long)

# Build a PyTorch Geometric Data object:
data = Data(
    x=node_features,  # NxF node feature matrix
    edge_index=edge_index,
    y=labels
)
# For convenience, assume this is our entire "dataset." 
# In practice, you might have multiple graphs or a larger multi-graph approach.

# We can wrap it in a simple list for DataLoader if we wanted mini-batches:
dataset = [data]
loader = DataLoader(dataset, batch_size=1)


################################################################################
# 2) DEFINE A SIMPLE GRAPH NEURAL NETWORK MODEL (e.g., GCN) FOR CLASSIFICATION
################################################################################

class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 1st GCN layer
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        # 2nd GCN layer
        x = self.gcn2(x, edge_index)
        return x

model = SimpleGCN(in_channels=fused_dim, hidden_channels=16, out_channels=2)

################################################################################
# 3) TRAIN THE MODEL
################################################################################

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()

num_epochs = 100
for epoch in range(num_epochs):
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)  # shape: [8, 2]
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

################################################################################
# 4) EVALUATE THE MODEL 
################################################################################

model.eval()
with torch.no_grad():
    # Again, this is a trivial single-graph dataset with 8 nodes:
    for batch in loader:
        logits = model(batch.x, batch.edge_index)  # shape [8, 2]
        preds = logits.argmax(dim=1)               # shape [8]
        correct = (preds == batch.y).sum()
        accuracy = correct.item() / preds.size(0)
        print(f"Predicted classes: {preds.tolist()}")
        print(f"True labels:       {batch.y.tolist()}")
        print(f"Accuracy: {accuracy*100:.2f}%")

################################################################################
# COMMENTS:
#  1) Here, "node_features" is a synthetic "late fusion" of imaging, text, 
#     and genomic data per node (patient).
#  2) 'edge_index' simulates the presence of clinically or biologically meaningful
#     relationships among patients (e.g., shared biomarkers).
#  3) We define a simple 2-layer GCN that tries to classify each node's label 
#     (e.g., 0 vs 1 for non-responder vs responder).
#  4) In a real scenario, you'd scale to bigger data, carefully craft the edges
#     from domain knowledge, handle multiple samples/graphs, and store data 
#     securely with appropriate governance and privacy measures.
################################################################################
