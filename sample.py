#!/usr/bin/env python3
"""
Sampling and evaluation script for DynaFlow with JAX/MuJoCo.

Supports:
- Trajectory sampling with ODE integration
- Evaluation metrics (MSE, TRE)
- Batch evaluation on test datasets
"""

import argparse
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from dyna_flow.model import ActionPredictor, count_parameters
from dyna_flow.rollout import create_go2_rollout
from dyna_flow.losses import sample_trajectory
from dyna_flow.data import load_trajectory_dataset, create_data_iterator
from dyna_flow.utils import load_checkpoint, compute_metrics, get_default_xml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample and evaluate DynaFlow")
    
    # Model
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA parameters if available")
    
    # MuJoCo
    parser.add_argument("--xml-path", type=str, default=None, help="Path to MuJoCo XML file")
    
    # Evaluation
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset")
    parser.add_argument("--eval-samples", type=int, default=32, help="Number of samples to evaluate")
    parser.add_argument("--batch", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Sampling
    parser.add_argument("--ode-steps", type=int, default=1, help="Number of ODE integration steps")
    parser.add_argument("--noise-scale", type=float, default=1.0, help="Scale of initial Gaussian noise")
    
    # Output
    parser.add_argument("--save-samples", type=str, default=None, help="Path to save generated samples")
    parser.add_argument("--verbose", action="store_true", help="Print detailed metrics")
    
    return parser.parse_args()


def evaluate_on_dataset(
    model,
    params,
    rollout_fn,
    dataset,
    batch_size: int,
    ode_steps: int,
    noise_scale: float,
    rng,
    verbose: bool = False
):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary of aggregated metrics
    """
    all_mses = []
    all_maes = []
    all_samples = []
    
    # Create iterator
    np_rng = np.random.default_rng(42)
    data_iter = create_data_iterator(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        rng=np_rng
    )
    
    for batch in tqdm(data_iter, desc="Evaluating", disable=not verbose):
        x_true = jnp.array(batch['trajectories'])  # (batch, H+1, state_dim)
        cond = jnp.array(batch['cond']) if 'cond' in batch else None
        
        # Extract initial states
        x0 = x_true[:, 0, :]
        
        # Sample trajectories
        rng, sample_rng = random.split(rng)
        x_sampled, u_sampled = sample_trajectory(
            model.apply,
            params,
            rollout_fn,
            x0,
            horizon=dataset.horizon,
            state_dim=dataset.state_dim,
            cond=cond,
            ode_steps=ode_steps,
            noise_scale=noise_scale,
            rng=sample_rng
        )
        
        # Compute metrics
        metrics = compute_metrics(x_sampled, x_true, exclude_first=True)
        all_mses.append(metrics['mse'])
        all_maes.append(metrics['mae'])
        
        if verbose:
            print(f"  MSE: {metrics['mse']:.6f}, MAE: {metrics['mae']:.6f}")
        
        all_samples.append({
            'x_pred': np.array(x_sampled),
            'x_true': np.array(x_true),
            'u_pred': np.array(u_sampled),
        })
    
    # Aggregate metrics
    results = {
        'mse_mean': float(np.mean(all_mses)),
        'mse_std': float(np.std(all_mses)),
        'mae_mean': float(np.mean(all_maes)),
        'mae_std': float(np.std(all_maes)),
        'num_samples': len(all_mses) * batch_size,
    }
    
    return results, all_samples


def main():
    args = parse_args()
    
    # Set random seed
    rng = random.PRNGKey(args.seed)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = load_checkpoint(args.ckpt)
    
    state_dim = ckpt['state_dim']
    action_dim = ckpt['action_dim']
    horizon = ckpt['horizon']
    cond_dim = ckpt.get('cond_dim', None)
    
    # Choose parameters (EMA or regular)
    if args.use_ema and 'ema_params' in ckpt:
        params = ckpt['ema_params']
        print("Using EMA parameters")
    else:
        params = ckpt['params']
        print("Using regular parameters")
    
    # Get XML path
    xml_path = args.xml_path
    if xml_path is None:
        try:
            xml_path = get_default_xml_path()
            print(f"Using default XML: {xml_path}")
        except FileNotFoundError:
            print("ERROR: Could not find Go2 XML file. Please provide --xml-path")
            return
    
    # Create rollout operator
    print("Creating MuJoCo rollout operator...")
    rollout = create_go2_rollout(xml_path=xml_path)
    
    # Create model
    print("Creating model...")
    model = ActionPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        cond_dim=cond_dim
    )
    print(f"Parameters: {count_parameters(params):,}")
    
    # Evaluation on dataset
    if args.dataset:
        print(f"\nLoading evaluation dataset from {args.dataset}")
        dataset = load_trajectory_dataset(
            paths=[args.dataset],
            horizon=horizon,
            stride=1
        )
        print(f"Loaded {len(dataset)} trajectory windows")
        
        # Limit to eval_samples
        if len(dataset) > args.eval_samples:
            print(f"Randomly selecting {args.eval_samples} samples for evaluation")
            # Subsample dataset
            indices = np.random.choice(len(dataset), args.eval_samples, replace=False)
            dataset.data = dataset.data[indices]
            if dataset.cond_data is not None:
                dataset.cond_data = dataset.cond_data[indices]
        
        print(f"\nEvaluating with ODE steps: {args.ode_steps}")
        results, samples = evaluate_on_dataset(
            model,
            params,
            rollout,
            dataset,
            batch_size=args.batch,
            ode_steps=args.ode_steps,
            noise_scale=args.noise_scale,
            rng=rng,
            verbose=args.verbose
        )
        
        print("\n" + "="*60)
        print("Evaluation Results:")
        print("="*60)
        print(f"MSE:  {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
        print(f"MAE:  {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
        print(f"Samples: {results['num_samples']}")
        print("="*60)
        
        # Save samples if requested
        if args.save_samples:
            os.makedirs(os.path.dirname(args.save_samples), exist_ok=True)
            np.savez(
                args.save_samples,
                **{f'sample_{i}': s for i, s in enumerate(samples)},
                results=results
            )
            print(f"\nSaved samples to {args.save_samples}")
    
    else:
        print("\nNo dataset provided. Use --dataset to evaluate on trajectories.")
        print("Example usage:")
        print(f"  python sample.py --ckpt {args.ckpt} --dataset path/to/test_data.npz --eval-samples 32")


if __name__ == "__main__":
    main()
