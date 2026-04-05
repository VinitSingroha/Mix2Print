"""
Evaluation metrics for bioink GNN challenge.
"""

import numpy as np
import pandas as pd
from typing import Dict

# Normalization ranges for NMAE
PRESSURE_RANGE = 1496.0  # kPa
TEMPERATURE_RANGE = 228.0  # °C
SPEED_RANGE = 90.0  # mm/s

# Difficulty-based weights for weighted NMAE
# Pressure is hardest to predict (highest weight), speed is easiest (lowest weight)
# Weights are empirically derived from participant score averages in leaderboard.csv
PRESSURE_WEIGHT = 0.60
TEMPERATURE_WEIGHT = 0.25
SPEED_WEIGHT = 0.15


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAE."""
    return float(np.mean(np.abs(y_true - y_pred)))


def normalized_mae(y_true: np.ndarray, y_pred: np.ndarray, value_range: float) -> float:
    """
    Calculate Normalized MAE.
    
    NMAE = MAE / range
    """
    mae = mean_absolute_error(y_true, y_pred)
    return mae / value_range


def compute_scores(predictions_path: str, ground_truth_path: str) -> Dict:
    """
    Compute all evaluation metrics.
    
    Returns dict with:
        - pressure_mae, temperature_mae, speed_mae
        - pressure_nmae, temperature_nmae, speed_nmae
        - combined_nmae (weighted average of three NMAEs)
        - combined_pct (combined_nmae as percentage)
        - n_samples
    """
    # Load data
    preds = pd.read_csv(predictions_path).sort_values('id')
    truth = pd.read_csv(ground_truth_path).sort_values('id')
    
    # Merge on ID
    merged = truth.merge(preds, on='id', how='inner', suffixes=('', '_pred'))
    
    if len(merged) != len(truth):
        raise ValueError(f"ID mismatch: expected {len(truth)} samples, got {len(merged)}")
    
    # Extract arrays (truth has _true suffix, preds don't)
    # Extract arrays based on column availability
    # Case 1: Truth has standard names (e.g. val.csv) -> Merge creates suffixes
    if 'pressure_pred' in merged.columns:
        pressure_true = merged['pressure'].values
        pressure_pred = merged['pressure_pred'].values
        
        temp_true = merged['temperature'].values
        temp_pred = merged['temperature_pred'].values
        
        speed_true = merged['speed'].values
        speed_pred = merged['speed_pred'].values
        
    # Case 2: Truth has _true names (e.g. test_labels.csv) -> No suffixes needed
    else:
        pressure_true = merged['pressure_true'].values
        pressure_pred = merged['pressure'].values
        
        temp_true = merged['temperature_true'].values
        temp_pred = merged['temperature'].values
        
        speed_true = merged['speed_true'].values
        speed_pred = merged['speed'].values
    
    # Compute MAE for each target
    pressure_mae = mean_absolute_error(pressure_true, pressure_pred)
    temperature_mae = mean_absolute_error(temp_true, temp_pred)
    speed_mae = mean_absolute_error(speed_true, speed_pred)
    
    # Compute NMAE for each target
    pressure_nmae = normalized_mae(pressure_true, pressure_pred, PRESSURE_RANGE)
    temperature_nmae = normalized_mae(temp_true, temp_pred, TEMPERATURE_RANGE)
    speed_nmae = normalized_mae(speed_true, speed_pred, SPEED_RANGE)
    
    # Combined NMAE (weighted based on difficulty)
    combined_nmae = (pressure_nmae * PRESSURE_WEIGHT) + (temperature_nmae * TEMPERATURE_WEIGHT) + (speed_nmae * SPEED_WEIGHT)
    
    return {
        'pressure_mae': float(pressure_mae),
        'temperature_mae': float(temperature_mae),
        'speed_mae': float(speed_mae),
        'pressure_nmae': float(pressure_nmae),
        'temperature_nmae': float(temperature_nmae),
        'speed_nmae': float(speed_nmae),
        'combined_nmae': float(combined_nmae),
        'combined_pct': float(combined_nmae * 100),
        'n_samples': len(merged)
    }
