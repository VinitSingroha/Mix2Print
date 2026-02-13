import torch
from torch_geometric.data import Data
import numpy as np
import os

def load_graph_dataset(graph_dir, is_train=True):
    """
    Loads .npy files into PyG Data objects.
    Returns: List of Data(x, edge_index, y)
    """
    files = os.listdir(graph_dir)
    # Filter for A files
    a_files = [f for f in files if f.endswith('_A.npy')]
    
    dataset = []
    
    for a_f in a_files:
        # graph_123_A.npy -> 123
        gid = int(a_f.split('_')[1])
        
        # Load Raw Matrices
        A = np.load(os.path.join(graph_dir, a_f))
        X = np.load(os.path.join(graph_dir, f'graph_{gid}_X.npy'))
        
        # Convert A to Edge Index (PyG)
        # Using pure connection if A[i,j] > 0
        edge_indices = np.where(A > 0)
        edge_index = torch.tensor(edge_indices, dtype=torch.long)
        
        # Convert X to Tensor
        x = torch.tensor(X, dtype=torch.float)
        
        # Target y
        y = None
        if is_train:
            y_data = np.load(os.path.join(graph_dir, f'graph_{gid}_y.npy'))
            y = torch.tensor(y_data, dtype=torch.float).view(1, -1)
            
        data = Data(x=x, edge_index=edge_index, y=y)
        data.gid = gid # Store ID for submission
        
        dataset.append(data)
        
    return dataset
