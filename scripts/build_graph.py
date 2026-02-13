import pandas as pd
import numpy as np
import os
import shutil
import sys

def build_compliant_graph_dataset():
    """
    Constructs the canonical Graph Dataset (A and X) strictly for NeurIPS compliance.
    
    Structure:
    data/public/train_graphs/
        graph_{id}_A.npy  [N x N] Adjacency (Binary, Fully Connected)
        graph_{id}_X.npy  [N x D] Node Features (One-Hot Identity + Concentration)
        graph_{id}_y.npy  [3]     Targets (Pressure, Temp, Speed)
        
    data/public/test_graphs/
        graph_{id}_A.npy
        graph_{id}_X.npy
        
    Feature Definition:
    - X covers 30 unique biomaterials (One-Hot) + 1 Concentration feature.
    - D = 31 dimensions.
    """
    print("Building NeurIPS-Compliant Graph Dataset (One-Hot + Conc)...")
    
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(ROOT)
    from competition.data_utils import parse_components
    
    # Paths
    train_path = os.path.join(ROOT, 'data/public/train.csv')
    test_feat_path = os.path.join(ROOT, 'data/public/test_features.csv')
    
    # Output Dirs
    train_out = os.path.join(ROOT, 'data/public/train_graphs')
    test_out = os.path.join(ROOT, 'data/public/test_graphs')
    
    if os.path.exists(train_out): shutil.rmtree(train_out)
    if os.path.exists(test_out): shutil.rmtree(test_out)
    os.makedirs(train_out)
    os.makedirs(test_out)

    # 1. Global Vocabulary (One-Hot Indexing)
    print("- Building Material Vocabulary...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_feat_path)
    
    all_materials = set()
    
    def gather_mats(df):
        for _, row in df.iterrows():
            comps = parse_components(row['Components'])
            for c in comps:
                all_materials.add(c['name'])

    gather_mats(df_train)
    gather_mats(df_test)
    
    material_list = sorted(list(all_materials))
    num_materials = len(material_list)
    mat_to_idx = {m: i for i, m in enumerate(material_list)}
    
    print(f"- Found {num_materials} unique materials.")
    print(f"- Node Feature Dim: {num_materials} (Identity) + 1 (Conc) = {num_materials + 1}")
    
    # Write vocabulary for reference
    with open(os.path.join(ROOT, 'data', 'public', 'node_vocabulary.txt'), 'w', encoding='utf-8') as f:
        for idx, m in enumerate(material_list):
            f.write(f"{idx},{m}\n")
    
    # 2. Process Graphs
    def process_split(df, out_dir, is_train=True):
        count = 0
        for _, row in df.iterrows():
            gid = row['id']
            comps = parse_components(row['Components'])
            
            n_nodes = len(comps)
            feature_dim = num_materials + 1
            
            # X: One-Hot + Concentration
            X = np.zeros((n_nodes, feature_dim), dtype=np.float32)
            
            # A: Binary Fully Connected
            A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
            
            for i, c in enumerate(comps):
                mat_idx = mat_to_idx[c['name']]
                conc = c['concentration']
                
                # One-Hot
                X[i, mat_idx] = 1.0
                # Concentration at last index
                X[i, -1] = conc
            
            # Binary Adjacency (Clique)
            # A_ij = 1 for all i,j (including self-loops? usually yes for GCN, no for some others)
            # NeurIPS Guideline says "Binary connectivity".
            # Standard: A_ij = 1 if connected. fully connected = all 1s.
            A.fill(1.0) 
            
            # Save
            np.save(os.path.join(out_dir, f'graph_{gid}_X.npy'), X)
            np.save(os.path.join(out_dir, f'graph_{gid}_A.npy'), A)
            
            if is_train:
                y = np.array([row['pressure'], row['temperature'], row['speed']], dtype=np.float32)
                np.save(os.path.join(out_dir, f'graph_{gid}_y.npy'), y)
                
            count += 1
        print(f"- Processed {count} graphs in {out_dir}")

    print("Processing Train...")
    process_split(df_train, train_out, is_train=True)
    
    print("Processing Test...")
    process_split(df_test, test_out, is_train=False)
    
    print("Done. Generated Compliant Dataset.")

if __name__ == "__main__":
    build_compliant_graph_dataset()
