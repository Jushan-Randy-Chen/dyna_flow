#!/usr/bin/env python3
"""
Analyze trajectory statistics to determine data-driven weight scales.

This script loads trajectory data and computes min/max/std for each state dimension
to help inform loss weighting decisions.
"""

import argparse
import numpy as np


def analyze_trajectories(npz_path: str):
    """Load and analyze trajectory statistics."""
    
    # Load data
    data = np.load(npz_path, allow_pickle=True)
    states = data['states']  # Shape: (N, H, state_dim)
    actions = data['actions']  # Shape: (N, H, action_dim)
    
    # Convert to float arrays if needed
    if states.dtype == object:
        states = np.array(states.tolist(), dtype=np.float32)
    if actions.dtype == object:
        actions = np.array(actions.tolist(), dtype=np.float32)
    
    print(f"Loaded trajectories from: {npz_path}")
    print(f"States shape: {states.shape}")
    print(f"Actions shape: {actions.shape}")
    print()
    
    # Flatten over batch and time dimensions
    states_flat = states.reshape(-1, states.shape[-1])  # (N*H, state_dim)
    
    # Compute statistics
    mins = states_flat.min(axis=0)
    maxs = states_flat.max(axis=0)
    means = states_flat.mean(axis=0)
    stds = states_flat.std(axis=0, ddof=1)
    ranges = maxs - mins
    
    # State dimension labels (37-dim state)
    labels = (
        ['pos_x', 'pos_y', 'pos_z'] +  # 0-2
        ['quat_w', 'quat_x', 'quat_y', 'quat_z'] +  # 3-6
        [f'joint_pos_{i}' for i in range(12)] +  # 7-18
        ['lin_vel_x', 'lin_vel_y', 'lin_vel_z'] +  # 19-21
        ['ang_vel_x', 'ang_vel_y', 'ang_vel_z'] +  # 22-24
        [f'joint_vel_{i}' for i in range(12)]  # 25-36
    )
    
    print("=" * 100)
    print(f"{'Dim':<4} {'Label':<15} {'Min':>10} {'Max':>10} {'Range':>10} {'Mean':>10} {'Std':>10} {'Inv_Var':>10}")
    print("=" * 100)
    
    for i, label in enumerate(labels):
        inv_var = 1.0 / (stds[i]**2 + 1e-6)
        print(f"{i:<4} {label:<15} {mins[i]:>10.4f} {maxs[i]:>10.4f} {ranges[i]:>10.4f} "
              f"{means[i]:>10.4f} {stds[i]:>10.4f} {inv_var:>10.4f}")
    
    print("=" * 100)
    print()
    
    # Group statistics by category
    print("=" * 100)
    print("GROUPED STATISTICS")
    print("=" * 100)
    
    groups = [
        ("Position (0-2)", slice(0, 3)),
        ("Quaternion (3-6)", slice(3, 7)),
        ("Joint Position (7-18)", slice(7, 19)),
        ("Linear Velocity (19-21)", slice(19, 22)),
        ("Angular Velocity (22-24)", slice(22, 25)),
        ("Joint Velocity (25-36)", slice(25, 37)),
    ]
    
    for name, idx in groups:
        group_stds = stds[idx]
        group_ranges = ranges[idx]
        print(f"\n{name}:")
        print(f"  Std range: [{group_stds.min():.4f}, {group_stds.max():.4f}]")
        print(f"  Mean std: {group_stds.mean():.4f}")
        print(f"  Value range: [{group_ranges.min():.4f}, {group_ranges.max():.4f}]")
    
    print()
    print("=" * 100)
    print("SUGGESTED WEIGHT SCALES (based on inverse variance)")
    print("=" * 100)
    
    # Compute normalized inverse variance weights
    inv_vars = 1.0 / (stds**2 + 1e-6)
    # Normalize so mean weight is 1.0
    normalized_weights = inv_vars / inv_vars.mean()
    
    print("\nPer-dimension weights (normalized, mean=1.0):")
    for i, label in enumerate(labels):
        print(f"  {i:2d} ({label:15s}): {normalized_weights[i]:8.4f}")
    
    print("\nGroup-averaged weights:")
    for name, idx in groups:
        group_weight = normalized_weights[idx].mean()
        print(f"  {name:<30s}: {group_weight:6.2f}")
    
    print()
    print("=" * 100)
    print("SUGGESTED SIMPLIFIED WEIGHT VECTOR")
    print("=" * 100)
    
    # Create simplified weight vector based on groups
    simplified_weights = np.ones(37)
    for name, idx in groups:
        simplified_weights[idx] = normalized_weights[idx].mean()
    
    print("\nNumPy array (copy-paste ready):")
    print(f"weight_scales = np.array([")
    for i in range(0, 37, 6):
        chunk = simplified_weights[i:i+6]
        formatted = ", ".join(f"{w:.4f}" for w in chunk)
        print(f"    {formatted},")
    print("])")
    
    print("\nPython list by groups:")
    print("weight_scales = np.array(")
    for name, idx in groups:
        weight = simplified_weights[idx][0]  # All same within group
        size = idx.stop - idx.start
        print(f"    [{'{'}{weight:.4f}{'}'}] * {size}  # {name}")
    print(")")
    
    # Also show alternative: inverse std (not squared)
    print()
    print("=" * 100)
    print("ALTERNATIVE: Inverse Std Weighting (not squared)")
    print("=" * 100)
    
    inv_stds = 1.0 / (stds + 1e-6)
    normalized_inv_stds = inv_stds / inv_stds.mean()
    
    print("\nGroup-averaged weights (inverse std):")
    for name, idx in groups:
        group_weight = normalized_inv_stds[idx].mean()
        print(f"  {name:<30s}: {group_weight:6.2f}")
    
    # Per-element weights with quaternion grouped
    print()
    print("=" * 100)
    print("PER-ELEMENT WEIGHTS (Quaternion Group-Averaged)")
    print("=" * 100)
    
    # Create per-element weights with quaternion averaged
    per_elem_weights = normalized_inv_stds.copy()
    quat_avg_weight = normalized_inv_stds[3:7].mean()
    per_elem_weights[3:7] = quat_avg_weight
    
    print("\nPer-element inverse std weights (quaternion group-averaged):")
    print("weights = np.array([")
    for i in range(0, 37, 5):
        chunk = per_elem_weights[i:min(i+5, 37)]
        formatted = ", ".join(f"{w:.4f}" for w in chunk)
        print(f"    {formatted},")
    print("])")
    
    print(f"\nMean weight: {per_elem_weights.mean():.4f}")
    print(f"Weight range: [{per_elem_weights.min():.4f}, {per_elem_weights.max():.4f}]")
    
    print("\nAll dimensions:")
    for i, label in enumerate(labels):
        print(f"  {i:2d} ({label:<14s}): std={stds[i]:.4f}, weight={per_elem_weights[i]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze trajectory statistics')
    parser.add_argument(
        '--data',
        type=str,
        default='logs/ppo_policy2/trajectories/trajectories.npz',
        help='Path to trajectories.npz file'
    )
    
    args = parser.parse_args()
    analyze_trajectories(args.data)
