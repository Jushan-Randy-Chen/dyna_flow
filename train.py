#!/usr/bin/env python3
"""
Training script for DynaFlow with JAX/MuJoCo.

Implements the training loop with:
- Conditional matching loss
- EMA for model parameters
- W&B logging
- Gradient accumulation
"""

import argparse
import time
import os
from typing import Optional
import numpy as np

import jax
import jax.numpy as jnp
from jax import random, grad, jit, value_and_grad
import optax
from tqdm.auto import tqdm

from dyna_flow.model import create_action_predictor, count_parameters
from dyna_flow.rollout import create_go2_rollout
from dyna_flow.losses import create_loss_fn
from dyna_flow.data import load_trajectory_dataset, create_data_iterator, prepare_batch_for_training, numpy_to_jax
from dyna_flow.utils import (
    save_checkpoint, create_optimizer, ExponentialMovingAverage,
    setup_wandb, print_model_summary, get_default_xml_path
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DynaFlow with JAX/MuJoCo")
    
    # Data
    parser.add_argument("--data", nargs="+", required=True, help="Paths to trajectory files/directories")
    parser.add_argument("--state-columns", nargs="*", default=None, help="CSV columns for state (if using CSV)")
    parser.add_argument("--horizon", type=int, default=16, help="Trajectory horizon H")
    
    # Model
    parser.add_argument("--d-model", type=int, default=384, help="Hidden dimension")
    parser.add_argument("--n-heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=3, help="Number of transformer blocks")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # MuJoCo
    parser.add_argument("--xml-path", type=str, default=None, help="Path to MuJoCo XML file")
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation timestep")
    parser.add_argument("--action-scale", type=float, default=0.3, help="Action scaling factor")
    
    # Training
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay (0 disables)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Checkpointing
    parser.add_argument("--save", type=str, default="checkpoints/dynaflow.pkl", help="Checkpoint save path")
    parser.add_argument("--save-interval", type=int, default=10, help="Save every N epochs")
    
    # Logging
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--wandb", type=str, choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", type=str, default="dynaflow-jax")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    
    return parser.parse_args()


def train_step(params, opt_state, batch, apply_fn, rollout_fn, optimizer, rng):
    """Single training step with gradient update."""
    
    # Create loss function
    def loss_fn(params):
        x0 = batch['x0']
        x1 = batch['x1']
        t = batch['t']
        cond = batch.get('cond', None)
        
        # Forward pass
        from dyna_flow.losses import conditional_matching_loss
        loss, aux = conditional_matching_loss(
            apply_fn,
            params,
            rollout_fn,
            x0,
            x1,
            t,
            cond=cond,
            rng=rng
        )
        return loss, aux
    
    # Compute gradients
    (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
    
    # Compute gradient norm for monitoring
    grad_norm = optax.global_norm(grads)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, aux, grad_norm


def main():
    args = parse_args()
    
    # Set random seeds
    rng = random.PRNGKey(args.seed)
    np_rng = np.random.default_rng(args.seed)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_trajectory_dataset(
        paths=args.data,
        horizon=args.horizon,
        stride=1,
        state_columns=args.state_columns
    )
    print(f"Loaded {len(dataset)} trajectory windows")
    print(f"State dim: {dataset.state_dim}, Horizon: {dataset.horizon}")
    if dataset.cond_dim is not None:
        print(f"Conditioning dim: {dataset.cond_dim}")
    
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
    rollout = create_go2_rollout(
        xml_path=xml_path,
        dt=args.dt,
        action_scale=args.action_scale
    )
    
    # Create model
    print("Creating model...")
    rng, init_rng = random.split(rng)
    model, params = create_action_predictor(
        state_dim=dataset.state_dim,
        action_dim=rollout.action_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        cond_dim=dataset.cond_dim,
        rng=init_rng
    )
    print_model_summary(params, "ActionPredictor")
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm
    )
    opt_state = optimizer.init(params)
    
    # EMA
    ema = None
    if args.ema_decay > 0:
        ema = ExponentialMovingAverage(decay=args.ema_decay)
        ema.initialize(params)
        print(f"Using EMA with decay {args.ema_decay}")
    
    # W&B logging
    wandb_run = None
    if args.wandb != "disabled":
        config = vars(args)
        config.update({
            'state_dim': dataset.state_dim,
            'action_dim': rollout.action_dim,
            'total_params': count_parameters(params),
            'dataset_size': len(dataset),
        })
        wandb_run = setup_wandb(
            config,
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"dynaflow-{time.strftime('%Y%m%d-%H%M%S')}",
            mode=args.wandb
        )
    
    # JIT compile training step
    train_step_jit = jit(lambda p, o, b, r: train_step(
        p, o, b, model.apply, rollout, optimizer, r
    ))
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        
        # Create data iterator
        data_iter = create_data_iterator(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            rng=np_rng
        )
        
        # Progress bar for batches
        pbar = tqdm(data_iter, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        
        for batch_np in pbar:
            # Prepare batch
            batch_np = prepare_batch_for_training(batch_np, np_rng, noise_scale=1.0)
            batch = numpy_to_jax(batch_np)
            
            # Split RNG
            rng, step_rng = random.split(rng)
            
            # Training step
            params, opt_state, loss, aux, grad_norm = train_step_jit(
                params, opt_state, batch, step_rng
            )
            
            # Update EMA
            if ema is not None:
                ema.update(params)
            
            # Accumulate metrics
            epoch_loss += float(loss)
            epoch_steps += 1
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'grad_norm': f"{grad_norm:.2f}"
            })
            
            # Log to W&B
            if wandb_run and global_step % args.log_interval == 0:
                try:
                    import wandb
                    wandb.log({
                        'train/loss': float(loss),
                        'train/mse': float(aux['mse']),
                        'train/grad_norm': float(grad_norm),
                        'epoch': epoch + 1,
                        'step': global_step,
                    }, step=global_step)
                except Exception as e:
                    print(f"W&B logging error: {e}")
        
        # Epoch summary
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        
        if wandb_run:
            try:
                import wandb
                wandb.log({
                    'train/epoch_loss': avg_loss,
                    'epoch': epoch + 1
                }, step=global_step)
            except Exception:
                pass
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                args.save,
                params=params,
                state_dim=dataset.state_dim,
                action_dim=rollout.action_dim,
                horizon=args.horizon,
                cond_dim=dataset.cond_dim,
                ema_params=ema.get_params() if ema else None,
                ema_decay=args.ema_decay if ema else None,
                epoch=epoch + 1,
                global_step=global_step
            )
    
    print(f"\nTraining complete! Final checkpoint saved to {args.save}")
    
    # Finish W&B
    if wandb_run:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
