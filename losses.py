"""
Flow matching loss functions for DynaFlow in JAX.

Implements the conditional matching loss and velocity field formulation
from the DynaFlow paper (Eqs. 4-5).
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Callable


def interpolate_xt(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    t: jnp.ndarray
) -> jnp.ndarray:
    """
    Straight-line interpolation for optimal transport path.
    
    X_t = (1 - t) * X_0 + t * X_1
    
    Args:
        x0: Base samples (batch, H+1, state_dim)
        x1: Target samples (batch, H+1, state_dim)
        t: Time values in [0, 1] (batch, 1)
    Returns:
        X_t: Interpolated trajectories (batch, H+1, state_dim)
    """
    # Broadcast t to match trajectory shape
    t = t[:, :, None]  # (batch, 1, 1)
    return (1.0 - t) * x0 + t * x1


def velocity_field(
    x_t: jnp.ndarray,
    x1_hat: jnp.ndarray,
    t: jnp.ndarray,
    eps: float = 1e-6
) -> jnp.ndarray:
    """
    Compute velocity field for optimal transport path (Eq. 5).
    
    u_θ(X_t, t) = 1/(1 - t) * (E[X_1 | X_t] - X_t)
    
    where E[X_1 | X_t] is approximated by the deterministic rollout X̂_1.
    
    Args:
        x_t: Current trajectory (batch, H+1, state_dim)
        x1_hat: Predicted terminal trajectory (batch, H+1, state_dim)
        t: Time (batch, 1)
        eps: Small constant to avoid division by zero
    Returns:
        Velocity field (batch, H+1, state_dim)
    """
    t = t[:, :, None]  # (batch, 1, 1)
    denominator = jnp.maximum(1.0 - t, eps)
    return (x1_hat - x_t) / denominator


def conditional_matching_loss(
    action_predictor_apply: Callable,
    params: dict,
    rollout_fn: Callable,
    x0: jnp.ndarray,
    x1_demo: jnp.ndarray,
    t: jnp.ndarray,
    cond: Optional[jnp.ndarray] = None,
    weight_mask: Optional[jnp.ndarray] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, dict]:
    """
    Conditional matching loss (Eq. 4 from paper).
    
    L(θ) = E[||W ⊙ (X̂_1 - X_1)||²]
    
    where:
    - X̂_1 = Φ(x_0, D_θ(X_t, c, t)) is the rolled-out prediction
    - X_1 is the expert demonstration
    - W is an element-wise weight mask
    
    Args:
        action_predictor_apply: Forward function of action predictor
        params: Model parameters
        rollout_fn: Rollout operator Φ: (x0, U) → X_1
        x0: Base samples (batch, H+1, state_dim)
        x1_demo: Expert trajectories (batch, H+1, state_dim)
        t: Time values (batch, 1)
        cond: Optional conditioning (batch, cond_dim)
        weight_mask: Optional element-wise weights (batch, H+1, state_dim)
        rng: Optional random key for dropout
    Returns:
        (loss, aux_dict) where aux_dict contains x1_hat for monitoring
    """
    # Interpolate to get X_t
    X_t = interpolate_xt(x0, x1_demo, t)  # (batch, H+1, state_dim)
    
    # Predict actions using action predictor
    U_hat = action_predictor_apply(
        params,
        X_t,
        t,
        cond=cond,
        deterministic=rng is None,  # Use dropout during training if RNG provided
        rngs={'dropout': rng} if rng is not None else {}
    )  # (batch, H, action_dim)
    
    # Rollout to get predicted terminal trajectory
    x0_initial = x0[:, 0, :]  # (batch, state_dim) - use initial state
    X1_hat = rollout_fn(x0_initial, U_hat)  # (batch, H+1, state_dim)
    
    # Default weight mask: uniform except ignore the initial state
    if weight_mask is None:
        weight_mask = jnp.ones_like(X1_hat)
        weight_mask = weight_mask.at[:, 0, :].set(0.0)
    
    # Compute weighted MSE loss
    diff = X1_hat - x1_demo
    weighted_diff_sq = weight_mask * (diff ** 2)
    loss = weighted_diff_sq.sum() / jnp.maximum(weight_mask.sum(), 1.0)
    
    # Auxiliary outputs for monitoring
    aux = {
        'x1_hat': X1_hat,
        'u_hat': U_hat,
        'mse': (diff ** 2).mean(),
        'weighted_mse': loss,
    }
    
    return loss, aux


def create_loss_fn(
    action_predictor_apply: Callable,
    rollout_fn: Callable,
    weight_first_state: bool = False
) -> Callable:
    """
    Create a loss function with fixed action predictor and rollout.
    
    Args:
        action_predictor_apply: Model.apply function
        rollout_fn: Rollout operator
        weight_first_state: Whether to include initial state in loss
    Returns:
        Loss function with signature (params, batch, rng) -> (loss, aux)
    """
    def loss_fn(params, batch, rng=None):
        if isinstance(batch, dict):
            x0 = batch['x0']
            x1_demo = batch['x1']
            t = batch['t']
            cond = batch.get('cond', None)
        else:
            # Assume batch is tuple (x0, x1, t) or (x0, x1, t, cond)
            if len(batch) == 3:
                x0, x1_demo, t = batch
                cond = None
            else:
                x0, x1_demo, t, cond = batch
        
        # Create weight mask if needed
        weight_mask = None
        if not weight_first_state:
            weight_mask = jnp.ones_like(x1_demo)
            weight_mask = weight_mask.at[:, 0, :].set(0.0)
        
        return conditional_matching_loss(
            action_predictor_apply,
            params,
            rollout_fn,
            x0,
            x1_demo,
            t,
            cond=cond,
            weight_mask=weight_mask,
            rng=rng
        )
    
    return loss_fn


def ode_step(
    x_t: jnp.ndarray,
    action_predictor_apply: Callable,
    params: dict,
    rollout_fn: Callable,
    t: jnp.ndarray,
    dt: float = 1.0,
    cond: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Single ODE integration step using Euler method.
    
    X_{t+dt} = X_t + dt * u_θ(X_t, t)
    
    Args:
        x_t: Current trajectory (batch, H+1, state_dim)
        action_predictor_apply: Model forward function
        params: Model parameters
        rollout_fn: Rollout operator
        t: Current time (batch, 1)
        dt: Step size
        cond: Optional conditioning
    Returns:
        Updated trajectory (batch, H+1, state_dim)
    """
    # Predict actions
    U_hat = action_predictor_apply(
        params,
        x_t,
        t,
        cond=cond,
        deterministic=True
    )
    
    # Rollout to get E[X_1 | X_t]
    x0 = x_t[:, 0, :]
    X1_hat = rollout_fn(x0, U_hat)
    
    # Compute velocity field
    u = velocity_field(x_t, X1_hat, t)
    
    # Euler step
    return x_t + dt * u


def sample_trajectory(
    action_predictor_apply: Callable,
    params: dict,
    rollout_fn: Callable,
    x0: jnp.ndarray,
    horizon: int,
    state_dim: int,
    cond: Optional[jnp.ndarray] = None,
    ode_steps: int = 1,
    noise_scale: float = 1.0,
    rng: Optional[jax.random.PRNGKey] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample trajectories using ODE integration from t=0 to t=1.
    
    Args:
        action_predictor_apply: Model forward function
        params: Model parameters
        rollout_fn: Rollout operator
        x0: Initial states (batch, state_dim)
        horizon: Trajectory horizon H
        state_dim: State dimension
        cond: Optional conditioning (batch, cond_dim)
        ode_steps: Number of Euler substeps (1 = single step)
        noise_scale: Scale of initial Gaussian noise
        rng: Random key for sampling initial noise
    Returns:
        (X_sampled, U_sampled) trajectories and actions
    """
    batch_size = x0.shape[0]
    
    # Sample initial Gaussian noise X_0
    if rng is not None:
        noise = jax.random.normal(rng, (batch_size, horizon + 1, state_dim))
    else:
        noise = jnp.zeros((batch_size, horizon + 1, state_dim))
    
    # Scale noise by per-sample std (similar to PyTorch implementation)
    X_cur = noise * noise_scale
    X_cur = X_cur.at[:, 0, :].set(x0)  # Fix initial state
    
    # Integrate ODE from t=0 to t=1
    t = jnp.zeros((batch_size, 1))
    dt = 1.0 / max(1, ode_steps)
    
    for _ in range(ode_steps):
        X_cur = ode_step(
            X_cur,
            action_predictor_apply,
            params,
            rollout_fn,
            t,
            dt=dt,
            cond=cond
        )
        t = jnp.minimum(t + dt, 1.0)
    
    # Final action prediction at t=1
    t_final = jnp.ones((batch_size, 1))
    U_sampled = action_predictor_apply(
        params,
        X_cur,
        t_final,
        cond=cond,
        deterministic=True
    )
    
    return X_cur, U_sampled
