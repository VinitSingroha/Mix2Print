
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from gnn_utils import load_graph_dataset
import pandas as pd
import numpy as np

# Standard GCN
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 3)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Final Prediction
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

def main():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'data/public/train_graphs')
    TEST_DIR = os.path.join(BASE_DIR, 'data/public/test_graphs')
    
    # Load
    print("Loading Data (GCN)...")
    train_dataset = load_graph_dataset(TRAIN_DIR, is_train=True)
    test_dataset = load_graph_dataset(TEST_DIR, is_train=False)
    
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    dim = train_dataset[0].x.shape[1]
    model = GCN(in_channels=dim, hidden_channels=64).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    crit = torch.nn.L1Loss()
    
    print("Training GCN...")
    for epoch in range(1, 101):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = crit(out, data.y)
            loss.backward()
            optimizer.step()
            loss_all += loss.item() * data.num_graphs
            
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss_all / len(train_dataset):.4f}')
            
    model.eval()
    preds = []
    ids = []
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
                    # Fallback if gid lost (should not happen if using proper PyG version)
                    ids.append(-1)
            
    preds = np.concatenate(preds, axis=0)
    df = pd.DataFrame(preds, columns=['pressure', 'temperature', 'speed'])
    df.insert(0, 'id', ids)
    
    os.makedirs('outputs', exist_ok=True)
    df.to_csv('outputs/gcn_submission.csv', index=False)
    print("Saved outputs/gcn_submission.csv")

if __name__ == "__main__":
    main()
