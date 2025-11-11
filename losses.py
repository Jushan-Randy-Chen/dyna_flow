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


def denormalize_states(x_norm: jnp.ndarray, stats: dict) -> jnp.ndarray:
    """Reverse z-score normalization.
    
    Args:
        x_norm: Normalized states (..., state_dim)
        stats: Dictionary with 'mean' and 'std'
    Returns:
        Original scale states


    """
    # Apply denormalization only to non-quaternion dims. Quaternion dims are
    # assumed to be left unchanged by normalization (stats mean=0,std=1) and
    # we still enforce unit norm for numerical safety.
    D = x_norm.shape[-1]
    idx = jnp.arange(D)
    non_quat_mask = jnp.logical_or(idx < 3, idx >= 7)

    # Denormalize selected channels
    x_denorm = x_norm
    sel = x_norm[..., non_quat_mask]
    mean_sel = stats['mean'][non_quat_mask]
    std_sel = stats['std'][non_quat_mask]
    sel_denorm = sel * std_sel + mean_sel
    x_denorm = x_denorm.at[..., non_quat_mask].set(sel_denorm)

    # Ensure quaternion component is unit length
    x_denorm = normalize_quat(x_denorm)

    return x_denorm


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
    std_sel = stats['std'][non_quat_mask]
    sel_norm = (sel - mean_sel) / std_sel
    x_norm = x_norm.at[..., non_quat_mask].set(sel_norm)

    # Enforce quaternion unit-norm for numeric stability
    x_norm = normalize_quat(x_norm)

    return x_norm


def normalize_quat(x: jnp.ndarray) -> jnp.ndarray:
    """Normalize quaternion component of state vector (indices 3:7).
    
    Args:
        x: State trajectories (..., state_dim)
    Returns:
        State with normalized quaternion
    """
    quat = x[..., 3:7]
    norm = jnp.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-6)
    return x.at[..., 3:7].set(quat / norm)


def conditional_matching_loss(
    action_predictor_apply: Callable,
    params: dict,
    rollout_fn: Callable,
    x0: jnp.ndarray,
    x1_demo: jnp.ndarray,
    t: jnp.ndarray,
    cond: Optional[jnp.ndarray] = None,
    weight_mask: Optional[jnp.ndarray] = None,
    norm_stats: Optional[dict] = None,
    dim_weights: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, dict]:
    """
    Conditional matching loss (Eq. 4 from paper).
    
    L(θ) = E[||(X̂_1 - X_1)||²] , assuming normalized states
    
    where:
    - X̂_1 = Φ(x_0, D_θ(X_t, c, t)) is the rolled-out prediction
    - X_1 is the expert demonstration
    - W is an element-wise weight mask
    
    Args:
        action_predictor_apply: Forward function of action predictor
        params: Model parameters
        rollout_fn: Rollout operator Φ: (x0, U) → X_1
        x0: Base samples (batch, H+1, state_dim) - NORMALIZED!
        x1_demo: Expert trajectories (batch, H+1, state_dim) - NORMALIZED!
        t: Time values (batch, 1)
        cond: NORMALIZED conditioning vector
        weight_mask: Optional element-wise weights (batch, H+1, state_dim)
        norm_stats: Normalization statistics dict with 'mean' and 'std' for denorm/renorm
        dim_weights: Optional per-dimension weights (state_dim,) for inverse std weighting
    Returns:
        (loss, aux_dict) where aux_dict contains x1_hat for monitoring
    """
    # Interpolate to get X_t
    X_t = interpolate_xt(x0, x1_demo, t)  # (batch, H+1, state_dim) -- normalized!
    
    # Predict actions using action predictor (deterministic, no dropout)
    """
    The model takes normalized states as input but outputs 
    actions in the raw action space that the simulator expects.
    Because the loss function compares predicted and demo trajectories, with the simulator acting as a bridge
    """
    U_hat = action_predictor_apply(
        params,
        X_t,
        t,
        cond=cond,
        deterministic=True
    )  # (batch, H, action_dim)
    
    # Rollout to get predicted terminal trajectory
    # NOTE: Rollout operates in RAW (unnormalized) space
    x0_initial = x0[:, 0, :]  # (batch, state_dim) - normalized initial state
    
    if norm_stats is not None:
        # Denormalize initial state for physics simulation
        x0_initial_raw = denormalize_states(x0_initial, norm_stats)
        # Run rollout in raw physical space
        X1_hat_raw = rollout_fn(x0_initial_raw, U_hat)  # (batch, H+1, state_dim)
        # Re-normalize for loss computation
        X1_hat = normalize_states(X1_hat_raw, norm_stats)
        X1_hat = normalize_quat(X1_hat)  # Re-normalize quaternions
    else:
        # No normalization stats provided - assume already in correct space
        X1_hat = rollout_fn(x0_initial, U_hat)  # (batch, H+1, state_dim)
    
    # Default weight mask:
    # - First state (t=0): weight 0.0 (preserved, no loss)
    # - Rest of states: weight 1.0
    if weight_mask is None:
        weight_mask = jnp.ones_like(X1_hat)
        weight_mask = weight_mask.at[:, 0, :].set(0.0)  # No loss on first state

    # if dim_weights is None:
    #     dim_weights_arr = jnp.ones((X1_hat.shape[-1],), dtype=X1_hat.dtype)
    # else:
    #     dim_weights_arr = jnp.asarray(dim_weights, dtype=X1_hat.dtype)

    # Special handling for quaternion components (indices 3:7):
    # Quaternions have sign ambiguity (q and -q represent same rotation).
    # To compute a consistent Euclidean-style difference (chordal distance)
    # we align signs by flipping predicted quaternions where the dot with
    # the demo quaternion is negative, then subtract component-wise.
    q_hat = X1_hat[..., 3:7]
    q_demo = x1_demo[..., 3:7]

    # Ensure unit length (numerical safety)
    q_hat = q_hat / jnp.linalg.norm(q_hat, axis=-1, keepdims=True).clip(min=1e-6)
    q_demo = q_demo / jnp.linalg.norm(q_demo, axis=-1, keepdims=True).clip(min=1e-6)

    dot = jnp.sum(q_hat * q_demo, axis=-1, keepdims=True)
    # If dot < 0, flip q_hat to maximize alignment
    sign = jnp.where(dot < 0.0, -1.0, 1.0)
    q_hat_aligned = q_hat * sign

    # Build aligned prediction array for subtraction
    X1_hat_aligned = X1_hat.at[..., 3:7].set(q_hat_aligned)

    # Compute quaternion angular error (scalar) and use it in the loss.
    # Align q_hat sign to q_demo to remove q ~ -q ambiguity (dot may be negative).
    q_hat = X1_hat_aligned[..., 3:7]
    q_demo = x1_demo[..., 3:7]
    # Ensure unit length (numerical safety)
    q_hat = q_hat / jnp.linalg.norm(q_hat, axis=-1, keepdims=True).clip(min=1e-6)
    q_demo = q_demo / jnp.linalg.norm(q_demo, axis=-1, keepdims=True).clip(min=1e-6)
    dot = jnp.sum(q_hat * q_demo, axis=-1, keepdims=True)
    dot_clipped = jnp.clip(dot, -1.0, 1.0)
    angle = 2.0 * jnp.arccos(dot_clipped)  # angle in radians
    angle_sq = angle ** 2  # scalar squared angle per sample/times

    # Prepare component-wise diff but zero quaternion components (we'll add angular term separately)
    diff = X1_hat_aligned - x1_demo
    diff_nonquat = diff.at[..., 3:7].set(0.0)

    weighted_mask = weight_mask
    weighted_diff_sq_nonquat = weighted_mask * (diff_nonquat ** 2) #non-quarternion component of error

    # Quaternion weight per sample/time: average weight across 4 quaternion components
    w_q = weight_mask[..., 3:7].mean(axis=-1)  # shape (batch, H+1)

    total_weight_nonquat = weighted_diff_sq_nonquat.sum()
    total_weight_quat = w_q.sum()

    quat_contrib = (w_q * angle_sq[..., 0]).sum()

    total_weight = total_weight_nonquat + total_weight_quat
    loss = (weighted_diff_sq_nonquat.sum() + quat_contrib) / jnp.maximum(total_weight, 1e-6)

    per_dim_weight = weighted_mask.sum(axis=(0, 1))
    per_dim_weighted_mse_nonquat = jnp.where(
        per_dim_weight > 0.0,
        weighted_diff_sq_nonquat.sum(axis=(0, 1)) / per_dim_weight,
        0.0,
    )
    
    # Quaternion weighted mse: broadcast scalar across quaternion components for diagnostics
    quat_weighted_mse_scalar = jnp.where(total_weight_quat > 0.0, quat_contrib / total_weight_quat, 0.0)
    per_dim_weighted_mse = per_dim_weighted_mse_nonquat.at[3:7].set(quat_weighted_mse_scalar)

    unweighted_mask = weight_mask.sum(axis=(0, 1))
    per_dim_unweighted_mse_nonquat = jnp.where(
        unweighted_mask > 0.0,
        (weight_mask * (diff_nonquat ** 2)).sum(axis=(0, 1)) / unweighted_mask,
        0.0,
    )
    quat_unweighted_mse_scalar = angle_sq.mean()
    per_dim_unweighted_mse = per_dim_unweighted_mse_nonquat.at[3:7].set(quat_unweighted_mse_scalar)

    # Auxiliary outputs for monitoring
    aux = {
        'x1_hat': X1_hat,
        'u_hat': U_hat,
        'mse': (diff ** 2).mean(),
        'weighted_mse': loss,
        'per_dim_weighted_mse': per_dim_weighted_mse,
        'per_dim_unweighted_mse': per_dim_unweighted_mse,
    }
    
    return loss, aux


def create_loss_fn(
    action_predictor_apply: Callable,
    rollout_fn: Callable,
    weight_first_state: bool = False,
    norm_stats: Optional[dict] = None,
    dim_weights: Optional[jnp.ndarray] = None,
) -> Callable:
    """
    Create a loss function with fixed action predictor and rollout.
    
    Args:
        action_predictor_apply: Model.apply function
        rollout_fn: Rollout operator
        weight_first_state: Whether to include initial state in loss
        norm_stats: Normalization statistics for denorm/renorm
        dim_weights: Per-dimension weights for inverse std weighting
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
            norm_stats=norm_stats,
            dim_weights=dim_weights,
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
    cond: Optional[jnp.ndarray] = None,
    norm_stats: Optional[dict] = None
) -> jnp.ndarray:
    """
    Single ODE integration step using Euler method.
    
    X_{t+dt} = X_t + dt * u_θ(X_t, t)
    
    Args:
        x_t: Current trajectory (batch, H+1, state_dim) - NORMALIZED
        action_predictor_apply: Model forward function
        params: Model parameters
        rollout_fn: Rollout operator
        t: Current time (batch, 1)
        dt: Step size
        cond: Optional conditioning
        norm_stats: Normalization statistics for denorm/renorm
    Returns:
        Updated trajectory (batch, H+1, state_dim) - NORMALIZED
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
    if norm_stats is not None:
        # Denormalize for rollout
        x0_raw = denormalize_states(x0, norm_stats)
        X1_hat_raw = rollout_fn(x0_raw, U_hat)
        # Re-normalize
        X1_hat = normalize_states(X1_hat_raw, norm_stats)
        X1_hat = normalize_quat(X1_hat)
    else:
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
    rng: Optional[jax.random.PRNGKey] = None,
    norm_stats: Optional[dict] = None
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample trajectories using ODE integration from t=0 to t=1.
    
    Args:
        action_predictor_apply: Model forward function
        params: Model parameters
        rollout_fn: Rollout operator
        x0: Initial states (batch, state_dim) - NORMALIZED if norm_stats provided
        horizon: Trajectory horizon H
        state_dim: State dimension
        cond: Optional conditioning (batch, cond_dim)
        ode_steps: Number of Euler substeps (1 = single step)
        noise_scale: Scale of initial Gaussian noise
        rng: Random key for sampling initial noise
        norm_stats: Normalization statistics for denorm/renorm
    Returns:
        (X_sampled, U_sampled) trajectories and actions - NORMALIZED if norm_stats provided
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
            cond=cond,
            norm_stats=norm_stats
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
