#!/usr/bin/env python3
"""
Train DynaFlow (dynamics-embedded flow matching) using JAX/Flax.

This implements the flow matching training pipeline from the DynaFlow paper
using JAX for automatic differentiation and parallel environment rollouts.
"""

import argparse
import os
import time
from functools import partial
from typing import Optional, Tuple, Any

# Configure JAX memory allocation BEFORE importing jax
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8' # Use only 80% of GPU memory

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax
from tqdm.auto import tqdm

# DynaFlow modules
from model import create_action_predictor, count_parameters
from rollout import MuJoCoGo2Rollout
from losses import conditional_matching_loss
from data import load_trajectory_dataset


def normalize_quat(x: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion component of state vector (indices 3:7)."""
    quat = x[..., 3:7]
    norm = jnp.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-6)
    return x.at[..., 3:7].set(quat / norm)


def compute_normalization_stats(trajectories: jnp.ndarray) -> dict:
    """Compute global mean and std per state dimension (matching ODE.py Normalizer).
    
    Args:
        trajectories: (N, H+1, state_dim) array of trajectory windows
    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (state_dim,)
    """
    # Compute mean/std over batch and time dimensions (0, 1), matching PyTorch Normalizer
    # PyTorch: x.mean(dim=(0,1)), x.std(dim=(0,1))
    mean = jnp.mean(trajectories, axis=(0, 1))
    std = jnp.std(trajectories, axis=(0, 1), ddof=1)  # Use Bessel's correction to match PyTorch
    std = jnp.clip(std, min=1e-6)  # Avoid division by zero
    
    return {'mean': mean, 'std': std}


def normalize_states(x: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Apply z-score normalization then renormalize quaternions.
    
    Args:
        x: State trajectories (..., state_dim)
        stats: Dictionary with 'mean' and 'std'
    Returns:
        Normalized states
    """
    # Apply z-score normalization
    x_norm = (x - stats['mean']) / stats['std']
    
    # Re-normalize quaternion component (indices 3:7)
    x_norm = normalize_quat(x_norm)
    
    return x_norm


def denormalize_states(x_norm: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Reverse z-score normalization (for logging/visualization).
    
    Args:
        x_norm: Normalized states (..., state_dim)
        stats: Dictionary with 'mean' and 'std'
    Returns:
        Original scale states
    """
    return x_norm * stats['std'] + stats['mean']


def create_train_state(
    rng: jax.Array,
    state_dim: int,
    action_dim: int,
    cond_dim: Optional[int],
    learning_rate: float,
    horizon: int,
    ema_decay: Optional[float] = None,
) -> Tuple[Any, Any, optax.GradientTransformation, optax.OptState, Optional[Any]]:
    """Create model and optimizer (matching ODE.py: constant LR, no schedule)."""
    
    # Initialize model 
    model, params = create_action_predictor(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=384,
        n_heads=6,
        depth=3, 
        cond_dim=cond_dim,
        rng=rng,
    )
    
    # Create optimizer with global norm clipping (matching ODE.py - constant LR, no schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Match ODE.py gradient clipping
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),  # Match ODE.py: constant lr=2e-4
    )
    opt_state = optimizer.init(params)
    
    # Initialize EMA parameters if requested
    ema_params = None
    if ema_decay is not None and ema_decay > 0.0:
        ema_params = jax.tree.map(lambda x: x.copy(), params)
    
    return model, params, optimizer, opt_state, ema_params


@partial(jax.jit, static_argnames=('model_fn', 'rollout_op', 'ema_decay', 'optimizer', 'use_dim_weights'))
def train_step(
    model_fn: Any,
    rollout_op: MuJoCoGo2Rollout,
    params: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x1_demo: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    rng: jax.Array,
    norm_stats: dict,
    dim_weights: Optional[jnp.ndarray] = None,
    use_dim_weights: bool = False,
    ema_params: Optional[Any] = None,
    ema_decay: Optional[float] = None,
) -> Tuple[Any, optax.OptState, float, float, jax.Array, Optional[Any], jnp.ndarray]:
    """Single training step with optional EMA update and per-dim MSE logging."""
    
    def loss_fn(params_in):
        # Normalize x1_demo
        x1_norm = normalize_states(x1_demo, norm_stats)
        
        # Sample x0 ~ N(0, I) for flow matching base distribution
        noise_key, t_key, dropout_key, cond_mask_key = random.split(rng, 4)
        
        # x0 is pure Gaussian noise (standard flow matching)
        x0 = random.normal(noise_key, x1_norm.shape)
        # Preserve first state exactly (conditioned on initial observation)
        x0 = x0.at[:, 0, :].set(x1_norm[:, 0, :])
        x0 = normalize_quat(x0)  # Re-normalize quaternions
        
        # # Apply 20% attribute dropout during training (matching ODE.py)
        # if cond is not None:
        #     # mask = (torch.rand(*condition.shape) > 0.2).int() from ODE.py
        #     cond_mask = (random.uniform(cond_mask_key, cond.shape) > 0.2).astype(jnp.float32)
        #     cond_masked = cond * cond_mask
        # else:
        #     cond_masked = None
        
        # Sample t ~ U(0,1)
        t = random.uniform(t_key, (x1_demo.shape[0], 1), minval=0.0, maxval=1.0)
        
        # Compute loss with dropout RNG and optional dimension weights
        loss, aux_dict = conditional_matching_loss(
            model_fn, params_in, rollout_op, x0, x1_norm, t, 
            cond=cond, dim_weights=dim_weights, rng=dropout_key
        )
        
        return loss, aux_dict
    
    # Compute gradients
    (loss, aux_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Extract x1_hat from aux dict and compute per-dimension MSE for diagnostics
    x1_norm = normalize_states(x1_demo, norm_stats)
    x1_hat = aux_dict['x1_hat']
    per_dim_mse = jnp.mean((x1_hat - x1_norm) ** 2, axis=(0, 1))  # Shape: (state_dim,)
    
    # Compute gradient norm for monitoring
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads))) #this is the pre-clipped grad norm
    
    # Update parameters (optimizer now includes gradient clipping)
    updates, opt_state = optimizer.update(grads, opt_state, params) 
    params = optax.apply_updates(params, updates)
    
    # Update EMA parameters if enabled
    if ema_params is not None and ema_decay is not None:
        ema_params = jax.tree.map(
            lambda ema, new: ema_decay * ema + (1.0 - ema_decay) * new,
            ema_params,
            params
        )
    
    return params, opt_state, loss, grad_norm, rng, ema_params, per_dim_mse


def main():
    parser = argparse.ArgumentParser(
        description="Train DynaFlow with JAX/Flax NNX"
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Paths to trajectory data (NPZ/NPY/CSV)",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default="Unitree_go2/go2_mjx_gym.xml",
        help="Path to MuJoCo XML model",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Trajectory horizon")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (ODE.py uses 2e-4)")
    parser.add_argument(
        "--stride",
        type=int,
        default=11, #ensures no overlap for horizon=10
        help="Stride for sliding window extraction (default: 10, reduces memory)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="logs/dynaflow_jax.npz",
        help="Model save path",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=50,
        help="Save checkpoint every N epochs (default: 50)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
        help="EMA decay rate (0 to disable)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Steps between logging",
    )
    parser.add_argument(
        "--state-columns",
        nargs="*",
        default=None,
        help="CSV state columns",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to load (for memory management)",
    )
    parser.add_argument(
        "--use-dim-weights",
        action="store_true",
        help="Enable inverse variance weighting per dimension (adaptive based on per_dim_mse)",
    )
    parser.add_argument(
        "--dim-weight-warmup",
        type=int,
        default=100,
        help="Number of steps before computing dimension weights (default: 100)",
    )
    
    # Weights & Biases
    parser.add_argument(
        "--wandb",
        type=str,
        choices=["online", "offline", "disabled"],
        default="disabled",
        help="W&B logging mode",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="dynaflow-jax",
        help="W&B project",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity",
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("DynaFlow Training with JAX/Flax NNX")
    print("=" * 80)
    print(f"Device: {jax.devices()}")
    print(f"Horizon: {args.horizon}")
    print(f"Batch size: {args.batch}")
    print(f"Learning rate: {args.lr}")
    print(f"EMA decay: {args.ema_decay if args.ema_decay > 0 else 'disabled'}")
    print("=" * 80)
    
    # Initialize RNG
    rng = random.PRNGKey(args.seed)
    
    # Load dataset
    print("\nLoading dataset...")
    max_windows = args.max_episodes * 100 if args.max_episodes else None  # ~50 windows per episode
    dataset = load_trajectory_dataset(
        paths=args.data,
        horizon=args.horizon,
        stride=args.stride,
        state_columns=args.state_columns,
        max_windows=max_windows,
    )
    
    state_dim = dataset.state_dim
    cond_dim = dataset.cond_dim
    n_samples = len(dataset)
    
    print(f"✓ Loaded {n_samples} trajectory windows")
    print(f"  State dim: {state_dim}")
    print(f"  Cond dim: {cond_dim}")
    print(f"  Horizon: {dataset.horizon}")
    
    # Pre-load entire dataset to GPU for faster training
    # This eliminates CPU→GPU transfer bottleneck on every training step
    # If you get OOM, reduce --max-episodes or --stride to limit dataset size
    print("\nTransferring dataset to GPU...")
    all_indices = np.arange(n_samples)
    all_data = dataset.get_batch(all_indices)
    all_trajectories = jnp.array(all_data['trajectories'])  # Transfer once to GPU
    all_conds = jnp.array(all_data['cond']) if 'cond' in all_data else None
    print(f"✓ Dataset on GPU: {all_trajectories.devices()}")
    print(f"  Memory: {all_trajectories.nbytes / 1e9:.2f} GB")
    
    # Compute normalization statistics (matching ODE.py)
    print("\nComputing normalization statistics...")
    norm_stats = compute_normalization_stats(all_trajectories)
    
    # Compute sigma_data: global std of normalized data (matching ODE.py)
    normalized_trajs = normalize_states(all_trajectories, norm_stats)
    sigma_data = float(jnp.std(normalized_trajs))
    norm_stats['sigma_data'] = sigma_data
    
    print(f"✓ Normalization stats computed")
    print(f"  Mean: min={float(norm_stats['mean'].min()):.3f}, max={float(norm_stats['mean'].max()):.3f}")
    print(f"  Std:  min={float(norm_stats['std'].min()):.3f}, max={float(norm_stats['std'].max()):.3f}")
    print(f"  sigma_data (global): {sigma_data:.3f}")
    
    # Setup rollout operator
    print("\nInitializing rollout operator...")
    rollout = MuJoCoGo2Rollout(xml_path=args.xml_path, dt=0.02)
    action_dim = rollout.action_dim
    print(f"✓ Rollout initialized (action_dim={action_dim})")
    
    # Create model and optimizer
    print("\nInitializing model...")
    rng, init_rng = random.split(rng)
    ema_decay = args.ema_decay if args.ema_decay > 0 else None
    
    model, params, optimizer, opt_state, ema_params = create_train_state(
        rng=init_rng,
        state_dim=state_dim,
        action_dim=action_dim,
        cond_dim=cond_dim,
        learning_rate=args.lr,
        horizon=args.horizon,
        ema_decay=ema_decay,
    )
    print(f"✓ Model initialized")
    print(f"  Parameters: {count_parameters(params)}")
    print(f"  Learning rate: {args.lr} (constant, matching ODE.py)")
    if ema_params is not None:
        print(f"  EMA enabled with decay={ema_decay}")
    
    # Setup W&B
    wandb_run = None
    if args.wandb != "disabled":
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb,
                config={
                    "horizon": args.horizon,
                    "batch": args.batch,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "ema_decay": args.ema_decay,
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                    "cond_dim": cond_dim,
                    "n_samples": n_samples,
                    "use_dim_weights": args.use_dim_weights,
                    "dim_weight_warmup": args.dim_weight_warmup,
                },
            )
            print("✓ W&B initialized")
        except Exception as e:
            print(f"W&B init failed: {e}")
    
    # Training loop
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    global_step = 0
    steps_per_epoch = n_samples // args.batch
    
    # Initialize dimension weights tracking
    dim_weights = None
    cumulative_per_dim_mse = None
    mse_count = 0
    
    if args.use_dim_weights:
        print(f"Inverse variance weighting enabled (warmup: {args.dim_weight_warmup} steps)")
    
    epoch_bar = tqdm(range(args.epochs), desc="Epochs")
    
    for epoch in epoch_bar:
        # Shuffle dataset
        rng, shuffle_rng = random.split(rng)
        indices = random.permutation(shuffle_rng, n_samples)
        
        epoch_loss = 0.0
        batch_bar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch+1}",
            leave=False,
        )
        
        for step in batch_bar:
            # Get batch - use GPU-resident data!
            batch_start = step * args.batch
            batch_end = batch_start + args.batch
            batch_indices = indices[batch_start:batch_end]
            
            # Index directly from GPU arrays (no CPU transfer!)
            x1_demo = all_trajectories[batch_indices]
            cond = all_conds[batch_indices] if all_conds is not None else None
            
            # Training step (note: normalization happens inside train_step now)
            rng, step_rng = random.split(rng)
            params, opt_state, loss, grad_norm, _, ema_params, per_dim_mse = train_step(
                model_fn=model.apply,
                rollout_op=rollout,
                params=params,
                opt_state=opt_state,
                optimizer=optimizer,
                x1_demo=x1_demo,
                cond=cond,
                rng=step_rng,
                norm_stats=norm_stats,
                dim_weights=dim_weights,
                use_dim_weights=args.use_dim_weights,
                ema_params=ema_params,
                ema_decay=ema_decay,
            )
            
            # Accumulate per-dimension MSE for computing weights
            if args.use_dim_weights:
                if cumulative_per_dim_mse is None:
                    cumulative_per_dim_mse = per_dim_mse
                else:
                    cumulative_per_dim_mse = cumulative_per_dim_mse + per_dim_mse
                mse_count += 1
                
                # Compute dimension weights after warmup period
                if global_step == args.dim_weight_warmup and mse_count > 0:
                    avg_per_dim_mse = cumulative_per_dim_mse / mse_count
                    # Inverse variance weighting: dimensions with higher MSE get lower weight
                    dim_weights = 1.0 / (avg_per_dim_mse + 1e-6)
                    # Normalize so mean weight is 1.0
                    dim_weights = dim_weights / dim_weights.mean()
                    print(f"\n✓ Dimension weights computed at step {global_step}")
                    print(f"  Weight range: [{float(dim_weights.min()):.3f}, {float(dim_weights.max()):.3f}]")
                    
                    # Log to W&B
                    if wandb_run:
                        try:
                            dim_weight_dict = {
                                f"dim_weights/dim_{i}": float(dim_weights[i])
                                for i in range(min(state_dim, 10))
                            }
                            wandb.log(dim_weight_dict, step=global_step)
                        except Exception:
                            pass
            
            # Convert JAX arrays to Python floats for logging
            loss = float(loss)
            grad_norm = float(grad_norm)
            
            epoch_loss += loss
            
            # Update progress
            avg_loss = epoch_loss / (step + 1)
            batch_bar.set_postfix(loss=f"{loss:.4f}", avg=f"{avg_loss:.4f}")
            
            # Log to W&B
            if wandb_run and global_step % args.log_interval == 0:
                try:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/avg_loss": avg_loss,
                            "train/grad_norm": grad_norm,
                            "epoch": epoch + 1,
                        },
                        step=global_step,
                    )
                    
                    # Log per-dimension MSE periodically (every 100 steps)
                    if global_step % 100 == 0:
                        per_dim_dict = {
                            f"train/mse_dim_{i}": float(per_dim_mse[i])
                            for i in range(min(state_dim, 10))  # Log first 10 dims to avoid clutter
                        }
                        wandb.log(per_dim_dict, step=global_step)
                except Exception:
                    pass
            
            global_step += 1
        
        # Epoch summary
        epoch_loss /= steps_per_epoch
        epoch_bar.set_postfix(loss=f"{epoch_loss:.4f}")
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f}")
        
        if wandb_run:
            try:
                wandb.log(
                    {"train/epoch_loss": epoch_loss, "epoch": epoch + 1},
                    step=global_step,
                )
            except Exception:
                pass
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = args.save.replace('.npz', f'_epoch{epoch+1}.npz')
            os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
            
            from flax import serialization
            
            # Save EMA parameters if available
            ckpt_params = ema_params if ema_params is not None else params
            model_bytes = serialization.to_bytes(ckpt_params)
            
            save_dict = {
                'model': model_bytes,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'horizon': args.horizon,
                'cond_dim': cond_dim if cond_dim is not None else -1,
                'epoch': epoch + 1,
                'norm_mean': np.array(norm_stats['mean']),
                'norm_std': np.array(norm_stats['std']),
                'sigma_data': norm_stats['sigma_data'],
            }
            
            if ema_params is not None:
                save_dict['model_regular'] = serialization.to_bytes(params)
                save_dict['ema_decay'] = ema_decay
            
            np.savez(checkpoint_path, **save_dict)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    # Save model
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    
    # Save using Flax serialization
    from flax import serialization
    
    # Use EMA parameters if available, otherwise use regular params
    final_params = ema_params if ema_params is not None else params
    model_bytes = serialization.to_bytes(final_params)
    
    # Save with metadata
    save_dict = {
        'model': model_bytes,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'horizon': args.horizon,
        'cond_dim': cond_dim if cond_dim is not None else -1,
        'norm_mean': np.array(norm_stats['mean']),
        'norm_std': np.array(norm_stats['std']),
        'sigma_data': norm_stats['sigma_data'],
    }
    
    # Also save regular params and EMA separately if EMA was used
    if ema_params is not None:
        save_dict['model_regular'] = serialization.to_bytes(params)
        save_dict['ema_decay'] = ema_decay
        print(f"  Saving EMA parameters (decay={ema_decay})")
    
    np.savez(args.save, **save_dict)
    
    print(f"✓ Model saved to {args.save}")
    
    if wandb_run:
        try:
            artifact = wandb.Artifact("dynaflow-model", type="model")
            artifact.add_file(args.save)
            wandb_run.log_artifact(artifact)
            wandb_run.finish()
        except Exception as e:
            print(f"W&B artifact upload failed: {e}")
    
    print("\nDone! 🎉")


if __name__ == "__main__":
    main()


"""
Usage Examples:
python train_flow_matching.py \
    --data logs/ppo_policy2/trajectories/trajectories.npz \
    --use-dim-weights \
    --dim-weight-warmup 100 \
    --wandb online \
    --wandb-project dynaflow-go2


Memory Management Tips:
- Use --stride 5-10 to reduce sliding window density
- Use --max-episodes to limit dataset size
- Reduce --batch to 32 or 64 if OOM
- Reduce --horizon to 8-10 for less memory per sample
- The script now limits JAX to 70% GPU memory by default
"""
