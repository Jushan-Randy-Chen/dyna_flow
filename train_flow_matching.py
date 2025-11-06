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
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use only 70% of GPU memory

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


def create_train_state(
    rng: jax.Array,
    state_dim: int,
    action_dim: int,
    cond_dim: Optional[int],
    learning_rate: float,
    horizon: int,
    ema_decay: Optional[float] = None,
) -> Tuple[Any, Any, optax.GradientTransformation, optax.OptState, Optional[Any]]:
    """Create model and optimizer."""
    
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
    
    # Create optimizer
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=1e-4)
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
    x1_demo: jnp.ndarray,
    cond: Optional[jnp.ndarray],
    rng: jax.Array,
    ema_params: Optional[Any] = None,
    ema_decay: Optional[float] = None,
) -> Tuple[Any, optax.OptState, float, float, jax.Array, Optional[Any]]:
    """Single training step with optional EMA update."""
    
    def loss_fn(params_in):
        # Sample x0 ~ p0 (perturbed expert with noise)
        std = x1_demo.std(axis=(1, 2), keepdims=True).clip(min=1e-3)
        noise_key, t_key, dropout_key = random.split(rng, 3)
        noise = random.normal(noise_key, x1_demo.shape) * std * 0.1
        x0 = x1_demo + noise
        x0 = x0.at[:, 0, :].set(x1_demo[:, 0, :])  # Keep initial state
        x0 = normalize_quat(x0)
        
        # Sample t ~ U(0,1)
        t = random.uniform(t_key, (x1_demo.shape[0], 1), minval=0.0, maxval=1.0)
        
        # Compute loss with dropout RNG
        loss, x1_hat = conditional_matching_loss(
            model_fn, params_in, rollout_op, x0, x1_demo, t, cond=cond, rng=dropout_key
        )
        
        return loss
    
    # Compute gradients
    loss, grads = jax.value_and_grad(loss_fn)(params)
    
    # Compute gradient norm for monitoring
    grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
    
    # Clip gradients
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # Update EMA parameters if enabled
    if ema_params is not None and ema_decay is not None:
        ema_params = jax.tree.map(
            lambda ema, new: ema_decay * ema + (1.0 - ema_decay) * new,
            ema_params,
            params
        )
    
    return params, opt_state, loss, grad_norm, rng, ema_params


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
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Stride for sliding window extraction (default: 5, reduces memory)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="logs/dynaflow_jax.npz",
        help="Model save path",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.995,
        help="EMA decay rate (0 to disable, typical: 0.995-0.9999)",
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
            # Get batch
            batch_start = step * args.batch
            batch_end = batch_start + args.batch
            batch_indices = indices[batch_start:batch_end]
            
            batch_data = dataset.get_batch(np.array(batch_indices))
            x1_demo = jnp.array(batch_data['trajectories'])
            cond = jnp.array(batch_data['cond']) if 'cond' in batch_data else None
            
            # Normalize quaternions
            x1_demo = normalize_quat(x1_demo)
            
            # Training step
            rng, step_rng = random.split(rng)
            params, opt_state, loss, grad_norm, _, ema_params = train_step(
                model_fn=model.apply,
                rollout_op=rollout,
                params=params,
                opt_state=opt_state,
                optimizer=optimizer,
                x1_demo=x1_demo,
                cond=cond,
                rng=step_rng,
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

# Basic training (memory-efficient defaults):
python train_flow_matching.py --data logs/ppo_policy2/trajectories/trajectories.npz

# Limit memory usage for large datasets:
python train_flow_matching.py \\
    --data logs/ppo_policy2/trajectories/trajectories.npz \\
    --max-episodes 5000 \\
    --stride 10 \\
    --batch 32

# With custom hyperparameters:
python train_flow_matching.py \\
    --data logs/ppo_policy2/trajectories/trajectories.npz \\
    --horizon 10 \\
    --batch 64 \\
    --epochs 500 \\
    --lr 3e-4

# With stronger EMA (more stable, smoother):
python train_flow_matching.py \\
    --data logs/ppo_policy2/trajectories/trajectories.npz \\
    --ema-decay 0.999

# Disable EMA:
python train_flow_matching.py \\
    --data logs/ppo_policy2/trajectories/trajectories.npz \\
    --ema-decay 0

# With W&B logging:
python train_flow_matching.py \\
    --data logs/ppo_policy2/trajectories/trajectories.npz \\
    --wandb online \\
    --wandb-project dynaflow-go2

# Multiple data sources:
python train_flow_matching.py \\
    --data logs/ppo_policy/trajectories logs/ppo_policy2/trajectories

Memory Management Tips:
- Use --stride 5-10 to reduce sliding window density
- Use --max-episodes to limit dataset size
- Reduce --batch to 32 or 64 if OOM
- Reduce --horizon to 8-10 for less memory per sample
- The script now limits JAX to 70% GPU memory by default
"""
