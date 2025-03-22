# Step 1: Prepare Dummy Data
"""
Let’s assume we have two modalities for each patient:
Text (clinical notes)
Gene Expression (a numeric vector)
"""

# Import modules
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Example dummy data
# Each entry has a text string, a numpy array for gene expression, and a label (e.g., cancer subtype).
dummy_data = [
    {
        "text": "Patient complains of persistent cough and moderate weight loss.",
        "genes": np.random.rand(50),  # 50-gene expression vector
        "label": 0
    },
    {
        "text": "Patient exhibits high fever and has a family history of lung cancer.",
        "genes": np.random.rand(50),
        "label": 1
    },
    {
        "text": "Patient shows improved appetite but has newly discovered lung nodule.",
        "genes": np.random.rand(50),
        "label": 0
    },
    # ... more records ...
]

class MultiModalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        genes = item["genes"]
        label = item["label"]
        return text, torch.tensor(genes, dtype=torch.float32), label

# Create dataset and dataloader
dataset = MultiModalDataset(dummy_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 2: Define the Late-Fusion Model
"""
We’ll use:

A Transformer-based encoder (e.g., a mini BERT-like model) for the text modality.

A simple fully connected network for the gene-expression vector.

A fusion layer that combines the Transformer output and the gene-expression output, followed by a final classification layer.

For simplicity, we’ll mock a tiny Transformer-like encoder. In a real project, you’d use a pretrained model (e.g., Hugging Face’s BERT).
"""

class MockTextTransformer(nn.Module):
    """A stub representing a Transformer-based text encoder (like BERT)."""
    def __init__(self, vocab_size=1000, embed_dim=32, hidden_dim=64):
        super(MockTextTransformer, self).__init__()
        # Embedding + a small linear "Transformer" block for demonstration
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, hidden_dim)
        
    def forward(self, token_ids):
        # In reality, you'd do attention, multi-head layers, etc.
        embedded = self.embedding(token_ids)  # [batch, seq_len, embed_dim]
        # We'll just take a mean pooling over the sequence dimension
        embedded_mean = embedded.mean(dim=1)  # [batch, embed_dim]
        output = torch.relu(self.linear(embedded_mean))  # [batch, hidden_dim]
        return output


class GeneExpressionEncoder(nn.Module):
    """Encode gene-expression vectors through a small MLP."""
    def __init__(self, input_dim=50, hidden_dim=32):
        super(GeneExpressionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
    def forward(self, x):
        # A single layer for demonstration
        return F.relu(self.fc1(x))


class LateFusionModel(nn.Module):
    def __init__(self, text_hidden_dim=64, gene_hidden_dim=32, num_classes=2):
        super(LateFusionModel, self).__init__()
        # Mock transformer and gene MLP
        self.text_encoder = MockTextTransformer(hidden_dim=text_hidden_dim)
        self.gene_encoder = GeneExpressionEncoder(hidden_dim=gene_hidden_dim)
        
        # Fusion layer: simply concatenate the encoded text + gene representations
        fused_input_dim = text_hidden_dim + gene_hidden_dim
        self.fusion = nn.Linear(fused_input_dim, num_classes)
        
    def forward(self, token_ids, gene_vec):
        text_features = self.text_encoder(token_ids)
        gene_features = self.gene_encoder(gene_vec)
        
        # Late-fusion: combine both representations
        fused = torch.cat([text_features, gene_features], dim=1)
        output = self.fusion(fused)  # [batch_size, num_classes]
        
        return output

# Step 3: Tokenize Text (Very Simplified)
'''
A real workflow would use a tokenizer (like BERT’s WordPiece or Byte-Pair Encoding). Here, we’ll mock a tokenizer by converting each character to an integer, just for demonstration.
'''
def mock_tokenize(text, max_len=16):
    # Convert each character to an integer (NOT real-world practice!)
    tokens = [ord(c) % 1000 for c in text[:max_len]]
    # Pad if necessary
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return tokens

# Step 4: Training Loop (Conceptual)
'''
We’ll illustrate a minimal training loop using our dummy data. The objective is a simple classification (2 classes).
'''
model = LateFusionModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Example training loop
model.train()
for epoch in range(2):  # Just 2 epochs for demo
    for text_batch, genes_batch, labels_batch in dataloader:
        
        # Mock tokenize
        token_ids_batch = [mock_tokenize(t) for t in text_batch]
        token_ids_batch = torch.tensor(token_ids_batch, dtype=torch.long)
        
        # Forward pass
        logits = model(token_ids_batch, genes_batch)
        loss = criterion(logits, torch.tensor(labels_batch, dtype=torch.long))
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")














