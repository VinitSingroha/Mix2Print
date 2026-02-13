import numpy as np
import os
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def load_data(graph_dir):
    """
    Loads graph data and flattens it for MLP.
    Method: Mean pooling of node features X.
    """
    files = os.listdir(graph_dir)
    # Filter for X files
    x_files = [f for f in files if f.endswith('_X.npy')]
    
    ids = []
    features = []
    targets = []
    
    print(f"Loading {len(x_files)} graphs from {graph_dir}...")
    
    for x_f in x_files:
        # Extract ID: graph_123_X.npy -> 123
        gid = int(x_f.split('_')[1])
        
        # Load X: [N_nodes, N_features]
        X = np.load(os.path.join(graph_dir, x_f))
        
        # Mean Pooling -> [N_features]
        x_pooled = np.mean(X, axis=0)
        
        ids.append(gid)
        features.append(x_pooled)
        
        # Load Target y if available (Train)
        y_path = os.path.join(graph_dir, f'graph_{gid}_y.npy')
        if os.path.exists(y_path):
            y = np.load(y_path)
            targets.append(y)
            
    return np.array(ids), np.array(features), np.array(targets) if targets else None

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(BASE_DIR, 'data/public/train_graphs')
    TEST_DIR = os.path.join(BASE_DIR, 'data/public/test_graphs')
    
    # 1. Load Data
    train_ids, X_train, y_train = load_data(TRAIN_DIR)
    test_ids, X_test, _ = load_data(TEST_DIR)
    
    # 2. Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train MLP
    # 3 targets: Pressure, Temp, Speed
    print("Training MLP Regressor...")
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # 4. Predict
    y_pred = model.predict(X_test_scaled)
    
    # 5. Save Submission
    submission = pd.DataFrame(y_pred, columns=['pressure', 'temperature', 'speed'])
    submission.insert(0, 'id', test_ids)
    
    os.makedirs('outputs', exist_ok=True)
    submission.to_csv('outputs/mlp_submission.csv', index=False)
    print("Saved outputs/mlp_submission.csv")

if __name__ == "__main__":
    main()
