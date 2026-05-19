# GNN-CB Frozen Prompt Template (v1.0) — Filled for Mix2Print

---

## SYSTEM PROMPT

```
Given a GNN coding competition consisting of:
  (1) the competition README,
  (2) the repository file tree,
  (3) a data-sample summary (file shapes, dtypes, head() of CSVs),
  (4) the required submission format.

Your task is to produce a single, self-contained Python script that, when
executed from the repository root in a CPU-only environment, trains a GNN,
generates predictions for the test set, and writes the submission file at the
location specified by the README.

You must respond in exactly this format:

<plan>
A 5-10 bullet plan covering: which features to use, model architecture,
training protocol, validation strategy, threshold/decoding, and how the
submission file will be written.
</plan>

<code>
A complete, runnable Python script. Constraints:
  - Use only the libraries declared in requirements.txt of the repo
    (plus the Python standard library).
  - The script must be runnable as `python solution.py` from the repo root.
  - Set seeds (numpy, torch, random) for reproducibility.
  - The model must be a Graph Neural Network (per competition rules).
  - The script must complete on CPU within 60 minutes for this competition.
  - The script must write the submission file in exactly the format the
    competition README specifies.
  - Do not download external data; use only files already on disk.
</code>

After the script is executed, you may receive an error trace. If so, return a
revised <plan> and <code> that addresses the specific error. You have at most
5 repair iterations.
```

---

## USER PROMPT (slots filled for Mix2Print)

```
## Competition README

# Mix2Print: Learning Material Interaction Physics for identifying parameters of 3D Bioprinting

## Task
Predict three continuous targets from bioink formulation graphs:
- Pressure (kPa): Extrusion force
- Temperature (°C): Printing temperature
- Speed (mm/s): Print head velocity

## Graph Specification
Each formulation is a graph G_i = (V_i, E_i, X_i) where:
- V_i: Biomaterials in formulation i
- E_i: Fully connected edges between all materials
- X_i: Node feature matrix, shape (n_i x D) where D ~ 31

Target y_i in R^3: (pressure, temperature, speed)

Files:
- data/public/train_graphs/graph_{id}_A.npy  — adjacency matrix
- data/public/train_graphs/graph_{id}_X.npy  — node features
- data/public/train_graphs/graph_{id}_y.npy  — targets (train only)
- data/public/test_graphs/graph_{id}_A.npy   — adjacency matrix
- data/public/test_graphs/graph_{id}_X.npy   — node features
- data/public/test_nodes.csv                 — test IDs to predict
- data/public/train.csv                      — training metadata

## Evaluation Metric
NMAE = (1/3) x [MAE_pressure/1496 + MAE_temperature/228 + MAE_speed/90]
Lower is better. Range: 0.0 (perfect) to 1.0+ (poor).
Baseline Random Forest NMAE = 0.060

## Dataset
- 423 formulations total
- 303 training / 120 test samples
- 30 biomaterials as node types

## Repository tree

data/public/
├── train_graphs/        ← graph_{id}_A.npy, graph_{id}_X.npy, graph_{id}_y.npy
├── test_graphs/         ← graph_{id}_A.npy, graph_{id}_X.npy
├── node_vocabulary.txt  ← material index mapping
├── sample_submission.csv
├── submission.key       ← encryption key
├── test_features.csv
├── test_nodes.csv       ← test IDs
└── train.csv            ← training data

## Data sample

### test_nodes.csv
- Shape: (120, ~2) — test graph IDs to predict
- Columns: id (graph id)
- IDs range from 340 to 399 (approximately)

### train.csv
- Shape: (303, ~35) — training formulations
- Contains biomaterial concentrations and printing parameters

### node_vocabulary.txt
- 30 biomaterial names, one per line, index = line number

### graph_{id}_X.npy
- Shape: (n_materials x 31) — one-hot material identity (30 dims) + concentration (1 dim)

### graph_{id}_A.npy
- Shape: (n_materials x n_materials) — fully connected binary adjacency matrix

### graph_{id}_y.npy
- Shape: (3,) — [pressure, temperature, speed]

## Required submission format

File: predictions.csv
Format:
  id,pressure,temperature,speed
  340,150.5,25.0,5.0
  341,800.0,155.0,1.2
  ...
  399,45.0,23.0,8.5

Requirements:
- One row per test graph id
- Exactly 120 rows (excluding header)
- id matches test_nodes.csv
- pressure in kPa, temperature in °C, speed in mm/s
- All values must be positive real numbers

## Allowed libraries

pandas>=2.0.0
numpy>=1.24.0
scipy
scikit-learn>=1.3.0

Note: torch and torch_geometric are NOT in requirements.txt.
Use numpy/scipy/sklearn to implement a GNN-style message passing manually,
OR use only the listed libraries for a graph-aware model.

Produce <plan> then <code>.
```
