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
from pathlib import Path
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
import orbax.checkpoint as ocp

# DynaFlow modules
from model import create_action_predictor, count_parameters, params_to_bytes
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
    # Exclude quaternion components from z-score statistics.
    # We compute mean/std over batch and time for non-quaternion dims and
    # set quaternion dims' mean=0, std=1 so they are left unchanged by z-score.
    # This preserves the unit-norm manifold structure of quaternions.
    # trajectories = jnp.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)
    D = trajectories.shape[-1]
    idx = jnp.arange(D)
    non_quat_mask = jnp.logical_or(idx < 3, idx >= 7)  # True for dims to normalize

    # Select non-quaternion channels and compute stats there
    sel = trajectories[..., non_quat_mask]
    mean_sel = jnp.mean(sel, axis=(0, 1))
    std_sel = jnp.std(sel, axis=(0, 1), ddof=0)
    # mean_sel = jnp.nan_to_num(mean_sel, nan=0.0)
    # std_sel = jnp.nan_to_num(std_sel, nan=1.0, posinf=1.0, neginf=1.0)
    std_sel = jnp.clip(std_sel, a_min=1e-6)
    
    # Build full mean/std arrays where quaternion dims have mean=0 and std=1
    mean = jnp.zeros((D,))
    std = jnp.ones((D,))
    mean = mean.at[non_quat_mask].set(mean_sel)
    std = std.at[non_quat_mask].set(std_sel)
    
    return {'mean': mean, 'std': std}


def compute_normalization_stats_cond(trajectories: jnp.ndarray) -> dict:
    """Compute mean/std for conditioning vectors (no quaternion handling).

    Args:
        trajectories: (N, H+1, cond_dim) array of conditioning windows
    Returns:
        Dictionary with 'mean' and 'std' arrays of shape (cond_dim,)
    """
    # trajectories = jnp.nan_to_num(trajectories, nan=0.0, posinf=0.0, neginf=0.0)
    mean = jnp.mean(trajectories, axis=(0, 1))
    std = jnp.std(trajectories, axis=(0, 1), ddof=0)
    # mean = jnp.nan_to_num(mean, nan=0.0)
    # std = jnp.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    std = jnp.clip(std, a_min=1e-6)
    return {'mean': mean, 'std': std}


def normalize_states(x: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Apply z-score normalization then renormalize quaternions.
    
    Args:
        x: State trajectories (..., state_dim)
        stats: Dictionary with 'mean' and 'std'
    Returns:
        Normalized states
    """
    # Apply z-score normalization only to non-quaternion dims. Quaternion dims
    # will be left unchanged (stats should have mean=0,std=1 for those dims).
    D = x.shape[-1]
    idx = jnp.arange(D)
    non_quat_mask = jnp.logical_or(idx < 3, idx >= 7)

    x_norm = x
    sel = x[..., non_quat_mask]
    mean_sel = stats['mean'][non_quat_mask]
    std_sel = jnp.clip(stats['std'][non_quat_mask], a_min=1e-6)
    sel_norm = (sel - mean_sel) / std_sel
    x_norm = x_norm.at[..., non_quat_mask].set(sel_norm)
    
    # Enforce quaternion unit-norm for numeric stability
    x_norm = normalize_quat(x_norm)
    return x_norm

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
        # NOTE: create_action_predictor expects the SEQUENCE LENGTH (H+1),
        # while this function receives the trajectory horizon H.
        # We add +1 here to include the initial state token.
        horizon=horizon + 1,
        cond_dim=cond_dim,
        rng=rng,
    )
    
    # Create optimizer with global norm clipping (matching ODE.py - constant LR, no schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(10.0),  # Match ODE.py gradient clipping
        optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),  
    )
    opt_state = optimizer.init(params)
    
    # Initialize EMA parameters if requested
    ema_params = None
    if ema_decay is not None and ema_decay > 0.0:
        ema_params = jax.tree.map(lambda x: x.copy(), params)
    
    return model, params, optimizer, opt_state, ema_params


@partial(jax.jit, static_argnames=('model_fn', 'rollout_op', 'ema_decay', 'optimizer'))
def train_step(
    model_fn: Any,
    rollout_op: MuJoCoGo2Rollout,
    params: Any,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    x1_demo: jnp.ndarray, #already normalized
    cond: Optional[jnp.ndarray], #already normalized
    rng: jax.Array,
    norm_stats: dict,
    ema_params: Optional[Any] = None,
    ema_decay: Optional[float] = None
) -> Tuple[Any, optax.OptState, float, float, jax.Array, Optional[Any], jnp.ndarray]:
    """Single training step with optional EMA update and per-dim MSE logging.
    
    Note: x1_demo and cond are expected to be ALREADY NORMALIZED.
    """
    
    def loss_fn(params_in):
        # x1_demo is already normalized
        x1_norm = x1_demo
        
        # Sample x0 ~ N(0, I) for flow matching base distribution
        noise_key, t_key = random.split(rng, 2)
        
        # x0 is pure Gaussian noise
        x0 = random.normal(noise_key, x1_norm.shape)
        # Preserve first state exactly (conditioned on initial observation)
        x0 = x0.at[:, 0, :].set(x1_norm[:, 0, :])
        x0 = normalize_quat(x0)  # ensure quaternion is unit quarternion

        t = random.uniform(t_key, (x1_demo.shape[0], 1), minval=0.0, maxval=1.0)
        # cond is already normalized (if present)
        cond_norm = cond
        
        # Compute loss
        loss, aux_dict = conditional_matching_loss(
            model_fn, params_in, rollout_op, x0, x1_norm, t, 
            cond=cond_norm, norm_stats=norm_stats
        )
        
        return loss, aux_dict
    
    # Compute gradients
    (loss, aux_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Extract per-dimension MSE aligned with training weights for diagnostics
    # per_dim_mse = aux_dict['per_dim_weighted_mse']  # Shape: (state_dim,)
    per_dim_mse = None
    
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
    parser.add_argument("--horizon", type=int, default=16, help="Trajectory horizon")
    parser.add_argument("--batch", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate (ODE.py uses 2e-4)")
    parser.add_argument(
        "--stride",
        type=int,
        default=17, #ensures no overlap for horizon=16
        help="Stride for sliding window extraction (default: 17, reduces memory)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="logs/dynaflow_jax.npz",
        help="Model save path",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="logs/checkpoints",
        help="Directory where Orbax checkpoints are stored",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of Orbax checkpoints to keep",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=25,
        help="Save checkpoint every N epochs (default: 25)",
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
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manager = ocp.CheckpointManager(
        str(checkpoint_dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=args.max_checkpoints,
            create=True,
        ),
    )
    
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
    
    state_dim = dataset.state_dim #37
    cond_dim = dataset.cond_dim #36
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

    print(f'trajectories has shape {all_trajectories.shape}, conditioning has shape {all_conds.shape}')
    print(f"✓ Dataset on GPU: {all_trajectories.devices()}")
    print(f"  Memory: {all_trajectories.nbytes / 1e9:.2f} GB")
    
    # Compute normalization statistics from raw data
    print("\nComputing normalization statistics...")
    norm_stats = compute_normalization_stats(all_trajectories)
    
    # print(f"✓ Normalization stats computed")
    # print(f"  Mean: min={float(norm_stats['mean'].min()):.3f}, max={float(norm_stats['mean'].max()):.3f}")
    # print(f"  Std:  min={float(norm_stats['std'].min()):.3f}, max={float(norm_stats['std'].max()):.3f}")

    # Conditioning normalization (if available)
    norm_stats_cond = None
    if all_conds is not None:
        # Conditioning is (N, H+1, cond_dim) - compute mean/std across batch and time
        norm_stats_cond = compute_normalization_stats_cond(all_conds)

    # Normalize dataset once
    print("\nNormalizing dataset...")
    all_trajectories = normalize_states(all_trajectories, norm_stats)
    if all_conds is not None:
        cond_std = jnp.clip(norm_stats_cond['std'], a_min=1e-6)
        all_conds = (all_conds - norm_stats_cond['mean']) / cond_std
    
    # Compute sigma_data from normalized data
    sigma_data = float(jnp.std(all_trajectories))
    norm_stats['sigma_data'] = sigma_data
    print(f"✓ Dataset normalized on GPU")
    print(f"  sigma_data (global std of normalized data): {sigma_data:.3f}")
    
    
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
            
            # Note: data is already normalized
            x1_demo = all_trajectories[batch_indices]
            cond = all_conds[batch_indices] if all_conds is not None else None
            
            # Training step
            rng, step_rng = random.split(rng)
    
            params, opt_state, loss, grad_norm, _, ema_params, _ = train_step(
                model_fn=model.apply,
                rollout_op=rollout,
                params=params,
                opt_state=opt_state,
                optimizer=optimizer,
                x1_demo=x1_demo,
                cond=cond,
                rng=step_rng,
                norm_stats=norm_stats,
                ema_params=ema_params,
                ema_decay=ema_decay,
            )
            
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
                    # if global_step % 100 == 0:
                    #     per_dim_dict = {
                    #         f"train/mse_dim_{i}": float(per_dim_mse[i])
                    #         for i in range(min(state_dim, 10))  # Log first 10 dims to avoid clutter
                    #     }
                    #     wandb.log(per_dim_dict, step=global_step)
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
            ckpt_params = ema_params if ema_params is not None else params
            state_payload = {
                "params": ckpt_params,
                "opt_state": opt_state,
                "norm_mean": norm_stats["mean"],
                "norm_std": norm_stats["std"],
            }
            if ema_params is not None:
                state_payload["ema_params"] = ema_params
            if norm_stats_cond is not None:
                state_payload["cond_norm_mean"] = norm_stats_cond["mean"]
                state_payload["cond_norm_std"] = norm_stats_cond["std"]
            metadata_payload = {
                "epoch": int(epoch + 1),
                "global_step": int(global_step),
                "state_dim": int(state_dim),
                "action_dim": int(action_dim),
                "horizon": int(args.horizon),
                "cond_dim": int(cond_dim) if cond_dim is not None else -1,
            }
            if ema_params is not None and ema_decay is not None:
                metadata_payload["ema_decay"] = float(ema_decay)
            if "sigma_data" in norm_stats:
                metadata_payload["sigma_data"] = float(norm_stats["sigma_data"])
            ckpt_args = ocp.args.Composite(
                state=ocp.args.StandardSave(state_payload),
                metadata=ocp.args.JsonSave(metadata_payload),
            )
            checkpoint_manager.save(epoch + 1, args=ckpt_args)
            checkpoint_manager.wait_until_finished()
            print(f"  ✓ Orbax checkpoint saved to {checkpoint_dir}")
    
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    
    # Save model
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)
    
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    
    # Use EMA parameters if available, otherwise use regular params
    final_params = ema_params if ema_params is not None else params
    model_bytes = params_to_bytes(final_params)
    
    # Save with metadata
    save_dict = {
        'model': model_bytes,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'horizon': args.horizon,
        'cond_dim': cond_dim if cond_dim is not None else -1,
        'norm_mean': np.array(norm_stats['mean']),
        'norm_std': np.array(norm_stats['std']),
        'cond_norm_mean': np.array(norm_stats_cond['mean']) if norm_stats_cond is not None else None,
        'cond_norm_std': np.array(norm_stats_cond['std']) if norm_stats_cond is not None else None,
    }
    
    # Also save regular params and EMA separately if EMA was used
    if ema_params is not None:
        save_dict['model_regular'] = params_to_bytes(params)
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
    --wandb online \
    --wandb-project dynaflow-go2


Memory Management Tips:
- Use --stride 5-10 to reduce sliding window density
- Use --max-episodes to limit dataset size
- Reduce --batch to 32 or 64 if OOM
- Reduce --horizon to 8-10 for less memory per sample
- The script now limits JAX to 80% GPU memory by default
"""
