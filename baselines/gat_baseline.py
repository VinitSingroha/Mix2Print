
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, GCNConv
from gnn_utils import load_graph_dataset
import pandas as pd
import numpy as np

# Use GATv2 for improved expressivity
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=3, heads=2):
        super(GAT, self).__init__()
        # Layer 1
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, dropout=0.2)
        
        # Layer 2 (Output)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.2)
        
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(out_channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )

    def forward(self, x, edge_index, batch):
        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        
        x = self.conv2(x, edge_index)
        
        # 2. Global Pooling
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # 3. Regression Head
        x = self.regressor(x)
        
        return x

def train_and_eval():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'data/public/train_graphs')
    TEST_DIR = os.path.join(BASE_DIR, 'data/public/test_graphs')
    
    # 1. Load Data (PyG Format)
    print("Loading Graph Datasets...")
    train_dataset = load_graph_dataset(TRAIN_DIR, is_train=True)
    test_dataset = load_graph_dataset(TEST_DIR, is_train=False)
    
    # 2. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 3. Initialize Model
    # Input Dim = 37 (from build_graph.py: 36 materials + 1 conc)
    # Check first graph to confirm dim
    input_dim = train_dataset[0].x.shape[1]
    print(f"Input Feature Dim: {input_dim}")
    
    model = GAT(in_channels=input_dim, hidden_channels=64, heads=4).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.L1Loss() # MAE is closer to NMAE metric
    
    # 4. Training Loop
    print("Start Training (GATv2)...")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y) # Standard L1
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            
        avg_loss = total_loss / len(train_dataset)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Loss: {avg_loss:.4f}")
            
    # 5. Predict on Test
    model.eval()
    ids = []
    preds = []
    
    print("Generating Predictions...")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = model(data.x, data.edge_index, data.batch)
            preds.append(out.cpu().numpy())
            # Handle potential tensor wrapping of custom attributes
            for d in data.to_data_list():
                if hasattr(d, 'gid'):
                    val = d.gid
                    if torch.is_tensor(val):
                        ids.append(val.item())
                    else:
                        ids.append(val)
                else:
                    ids.append(-1)
            
    preds = np.concatenate(preds, axis=0)
    
    # 6. Save Submission
    # Ensure IDs match order
    # Our loader might scramble if shuffle=True, but test_loader is shuffle=False.
    # However, PyG DataLoader doesn't preserve order perfectly across workers unless carefully managed.
    # We extracted IDs directly from the batch, so match is guaranteed.
    
    df = pd.DataFrame(preds, columns=['pressure', 'temperature', 'speed'])
    df.insert(0, 'id', ids)
    
    os.makedirs('outputs', exist_ok=True)
    out_path = 'outputs/gat_submission.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    train_and_eval()
