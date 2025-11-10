"""
Utility functions for DynaFlow: checkpointing, EMA, normalization, etc.
"""

import jax
import jax.numpy as jnp
import pickle
import numpy as np
import os
from typing import Any, Dict, Optional, Tuple
from flax.training import checkpoints
from flax import serialization
import optax


def save_checkpoint(
    save_path: str,
    params: dict,
    state_dim: int,
    action_dim: int,
    horizon: int,
    cond_dim: Optional[int] = None,
    ema_params: Optional[dict] = None,
    ema_decay: Optional[float] = None,
    **kwargs
):
    """
    Save model checkpoint.
    
    Args:
        save_path: Path to save checkpoint
        params: Model parameters
        state_dim: State dimension
        action_dim: Action dimension
        horizon: Trajectory horizon
        cond_dim: Optional conditioning dimension
        ema_params: Optional EMA parameters
        ema_decay: Optional EMA decay rate
        **kwargs: Additional metadata to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'params': params,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'horizon': horizon,
    }
    
    if cond_dim is not None:
        checkpoint['cond_dim'] = cond_dim
    
    if ema_params is not None:
        checkpoint['ema_params'] = ema_params
        checkpoint['ema_decay'] = ema_decay
    
    checkpoint.update(kwargs)
    
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(load_path: str) -> dict:
    """
    Load model checkpoint.
    
    Args:
        load_path: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint data
    """
    with open(load_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"Checkpoint loaded from {load_path}")
    return checkpoint


def load_npz_checkpoint(
    checkpoint_path: str,
    use_ema: bool = True
) -> Tuple[Dict[str, Any], Any]:
    """
    Load checkpoint from NPZ file (as saved by train_flow_matching.py).
    
    Args:
        checkpoint_path: Path to .npz checkpoint file
        use_ema: Whether to use EMA parameters (if available)
    
    Returns:
        metadata: Dict with state_dim, action_dim, horizon, cond_dim
        params: Deserialized model parameters (as Flax pytree)
    
    Example:
        >>> metadata, params = load_npz_checkpoint('logs/model_epoch50.npz')
        >>> # Create model with metadata
        >>> model, init_params = create_action_predictor(
        ...     state_dim=metadata['state_dim'],
        ...     action_dim=metadata['action_dim'],
        ...     cond_dim=metadata['cond_dim'],
        ...     rng=jax.random.PRNGKey(0)
        ... )
        >>> # params now contains trained weights ready for inference
    """
    # Load NPZ file
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    
    # Extract metadata
    metadata = {
        'state_dim': int(checkpoint['state_dim']),
        'action_dim': int(checkpoint['action_dim']),
        'horizon': int(checkpoint['horizon']),
        'cond_dim': int(checkpoint['cond_dim']) if checkpoint['cond_dim'] != -1 else None,
    }
    
    if 'epoch' in checkpoint:
        metadata['epoch'] = int(checkpoint['epoch'])
    
    # Choose which parameters to load
    if use_ema and 'model' in checkpoint:
        model_bytes = checkpoint['model'].tobytes()
        param_type = 'EMA' if 'ema_decay' in checkpoint else 'regular'
    elif 'model_regular' in checkpoint:
        model_bytes = checkpoint['model_regular'].tobytes()
        param_type = 'regular'
    elif 'model' in checkpoint:
        model_bytes = checkpoint['model'].tobytes()
        param_type = 'saved'
    else:
        raise ValueError("No model parameters found in checkpoint!")
    
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  State dim: {metadata['state_dim']}")
    print(f"  Action dim: {metadata['action_dim']}")
    print(f"  Horizon: {metadata['horizon']}")
    print(f"  Cond dim: {metadata['cond_dim']}")
    if 'epoch' in metadata:
        print(f"  Epoch: {metadata['epoch']}")
    print(f"  Using {param_type} parameters")
    
    # Return metadata and raw bytes (user must deserialize with model structure)
    metadata['model_bytes'] = model_bytes
    
    return metadata, model_bytes


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) for model parameters.
    
    Implements Polyak averaging: θ_ema = decay * θ_ema + (1 - decay) * θ
    """
    
    def __init__(self, decay: float = 0.995):
        """
        Args:
            decay: EMA decay rate (typical values: 0.999, 0.9999)
        """
        self.decay = decay
        self.ema_params = None
    
    def initialize(self, params: dict):
        """Initialize EMA parameters from model parameters."""
        self.ema_params = jax.tree.map(lambda x: x.copy(), params)
    
    def update(self, params: dict):
        """Update EMA parameters with new model parameters."""
        if self.ema_params is None:
            self.initialize(params)
        else:
            self.ema_params = jax.tree.map(
                lambda ema, new: self.decay * ema + (1 - self.decay) * new,
                self.ema_params,
                params
            )
    
    def get_params(self) -> dict:
        """Get current EMA parameters."""
        return self.ema_params


def create_optimizer(
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    warmup_steps: int = 0,
    max_grad_norm: float = 1.0
) -> optax.GradientTransformation:
    """
    Create AdamW optimizer with optional warmup and gradient clipping.
    
    Args:
        learning_rate: Peak learning rate
        weight_decay: L2 regularization weight
        warmup_steps: Number of linear warmup steps
        max_grad_norm: Maximum gradient norm for clipping
    
    Returns:
        Optax optimizer
    """
    # Learning rate schedule
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=100000,  # Large value for constant LR after warmup
            end_value=learning_rate
        )
    else:
        schedule = learning_rate
    
    # Chain optimizer components
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    )
    
    return optimizer


def compute_metrics(
    x_pred: jnp.ndarray,
    x_true: jnp.ndarray,
    exclude_first: bool = True
) -> Dict[str, float]:
    """
    Compute evaluation metrics for trajectory prediction.
    
    Args:
        x_pred: Predicted trajectories (batch, H+1, state_dim)
        x_true: Ground truth trajectories (batch, H+1, state_dim)
        exclude_first: Whether to exclude initial state from metrics
    
    Returns:
        Dictionary of metrics
    """
    if exclude_first:
        x_pred = x_pred[:, 1:, :]
        x_true = x_true[:, 1:, :]
    
    # Mean Squared Error
    mse = jnp.mean((x_pred - x_true) ** 2)
    
    # Mean Absolute Error
    mae = jnp.mean(jnp.abs(x_pred - x_true))
    
    # Per-timestep error
    per_step_mse = jnp.mean((x_pred - x_true) ** 2, axis=(0, 2))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'final_step_mse': float(per_step_mse[-1]),
        'mean_per_step_mse': float(per_step_mse.mean()),
    }


def normalize_state(
    state: jnp.ndarray,
    mean: Optional[jnp.ndarray] = None,
    std: Optional[jnp.ndarray] = None
) -> tuple:
    """
    Normalize state trajectories to zero mean and unit variance.
    
    Args:
        state: State trajectories (N, H+1, state_dim) or (H+1, state_dim)
        mean: Optional pre-computed mean
        std: Optional pre-computed std
    
    Returns:
        (normalized_state, mean, std)
    """
    if mean is None:
        mean = jnp.mean(state, axis=tuple(range(state.ndim - 1)))
    if std is None:
        std = jnp.std(state, axis=tuple(range(state.ndim - 1)))
        std = jnp.maximum(std, 1e-6)
    
    normalized = (state - mean) / std
    return normalized, mean, std


def denormalize_state(
    normalized_state: jnp.ndarray,
    mean: jnp.ndarray,
    std: jnp.ndarray
) -> jnp.ndarray:
    """Denormalize state trajectories."""
    return normalized_state * std + mean


def count_parameters(params: dict) -> int:
    """Count total number of parameters in model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def print_model_summary(params: dict, model_name: str = "Model"):
    """Print summary of model parameters."""
    total_params = count_parameters(params)
    print(f"\n{model_name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Size: {total_params * 4 / 1e6:.2f} MB (float32)")


def setup_wandb(config: dict, project: str = "dynaflow", entity: Optional[str] = None, **kwargs):
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Configuration dictionary to log
        project: W&B project name
        entity: W&B entity (team) name
        **kwargs: Additional arguments for wandb.init
    
    Returns:
        wandb run object or None if import fails
    """
    try:
        import wandb
        run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            **kwargs
        )
        return run
    except ImportError:
        print("Warning: wandb not installed. Skipping W&B logging.")
        return None
    except Exception as e:
        print(f"Warning: Failed to initialize W&B: {e}")
        return None


def get_default_xml_path() -> str:
    """Get default path to Go2 MuJoCo XML file."""
    # Try local Unitree_go2 folder first, then fall back to DDAT paths
    possible_paths = [
        "Unitree_go2/scene_mjx_gym.xml",
        os.path.join(os.path.dirname(__file__), "Unitree_go2/scene_mjx_gym.xml"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        "Could not find Go2 XML model. Please provide xml_path explicitly."
    )
