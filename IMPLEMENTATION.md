# DynaFlow Implementation Details

This document provides technical details about the JAX/MuJoCo implementation of DynaFlow.

## Architecture Overview

### 1. Action Predictor (DiT1d)

**File**: `model.py`

The action prediction network `D_θ` is implemented as a 1D Diffusion Transformer:

```
Input: X_t (batch, H+1, state_dim), t (batch, 1), c (batch, cond_dim)
  ↓
  Embed states → d_model dimension
  ↓
  Add sinusoidal positional encodings
  ↓
  Embed time t → MLP(t) 
  ↓
  (Optional) Embed conditioning c → Attention(c) + MLP
  ↓
  Add time + conditioning embeddings
  ↓
  Apply N DiT blocks with AdaLN-Zero
  ↓
  Final layer with AdaLN modulation
  ↓
  Project to action dimension
  ↓
Output: U^ (batch, H, action_dim)
```

**Key Components**:
- **DiTBlock**: Self-attention + MLP with adaptive layer norm modulation
- **TimeEmbedding**: Maps scalar t to embedding via MLP
- **ContinuousCondEmbedder**: Processes conditioning vectors with attention
- **AdaLN-Zero**: Adaptive layer norm with scale, shift, and gate parameters

### 2. Differentiable Rollout (MuJoCo MJX)

**File**: `rollout.py`

The rollout operator Φ: (x₀, U) → X₁ uses MuJoCo MJX for GPU-accelerated physics:

```python
def rollout(x0, U):
    # x0: (batch, state_dim)
    # U: (batch, H, action_dim)
    
    # Initialize MJX data
    data = mjx.make_data(model)
    data = set_state(data, x0)
    
    # Rollout loop
    X = [x0]
    for t in range(H):
        # PD control
        motor_targets = default_pose + U[:, t] * action_scale
        motor_targets = clip(motor_targets, joint_lowers, joint_uppers)
        
        # Step physics (differentiable!)
        data = data.replace(ctrl=motor_targets)
        data = mjx.step(model, data)
        
        # Extract state
        x_next = extract_state(data)
        X.append(x_next)
    
    return stack(X)  # (batch, H+1, state_dim)
```

**State Representation** (37D):
```
[base_pos(3), base_quat(4), joint_pos(12),
 base_linvel(3), base_angvel(3), joint_vel(12)]
```

**Action Representation** (12D):
```
Residual joint position targets from default pose
[FR_hip, FR_thigh, FR_calf,  # Front Right
 FL_hip, FL_thigh, FL_calf,  # Front Left
 RR_hip, RR_thigh, RR_calf,  # Rear Right
 RL_hip, RL_thigh, RL_calf]  # Rear Left
```

### 3. Flow Matching Loss

**File**: `losses.py`

**Conditional Matching Loss** (Eq. 4 from paper):
```
L(θ) = E[||W ⊙ (X̂₁ - X₁)||²]
```

**Implementation**:
```python
def conditional_matching_loss(params, x0, x1_demo, t, cond):
    # 1. Interpolate: X_t = (1-t)X_0 + tX_1
    X_t = interpolate(x0, x1_demo, t)
    
    # 2. Predict actions: U^ = D_θ(X_t, t, c)
    U_hat = action_predictor(params, X_t, t, cond)
    
    # 3. Rollout: X̂_1 = Φ(x_0, U^)
    X1_hat = rollout(x0[:, 0], U_hat)
    
    # 4. Compute loss with weight mask
    diff = X1_hat - x1_demo
    loss = (weight_mask * diff²).sum() / weight_mask.sum()
    
    return loss
```

**Velocity Field** (Eq. 5):
```
u_θ(X_t, t) = (E[X_1|X_t] - X_t) / (1 - t)
```

### 4. ODE Sampling

**Single-step ODE** (for t=0 to t=1):
```python
def sample(x0, params, rollout):
    # Initialize with Gaussian noise
    X_0 = x0 + noise
    t = 0.0
    
    # Single Euler step with dt=1
    U_hat = D_θ(X_0, t=0)
    X1_hat = Φ(x0, U_hat)
    u = (X1_hat - X_0) / 1.0
    X_1 = X_0 + 1.0 * u
    
    return X_1
```

**Multi-step ODE** (for better accuracy):
```python
dt = 1.0 / N_steps
X_t = X_0
for i in range(N_steps):
    U_hat = D_θ(X_t, t)
    X1_hat = Φ(x0, U_hat)
    u = (X1_hat - X_t) / (1 - t)
    X_t = X_t + dt * u
    t = t + dt
```

## Training Details

### Optimizer

AdamW with:
- Learning rate: 2×10⁻⁴
- Weight decay: 1×10⁻⁴
- Gradient clipping: max_norm = 1.0

### EMA (Exponential Moving Average)

Polyak averaging with decay = 0.995:
```
θ_ema ← 0.995 × θ_ema + 0.005 × θ
```

### Data Sampling

Each training iteration:
1. Sample expert trajectory X₁ ~ dataset
2. Sample time t ~ U(0, 1)
3. Sample noise X₀ ~ N(0, σ²) where σ = std(X₁)
4. Interpolate X_t = (1-t)X₀ + tX₁
5. Compute loss and backprop

### Weight Mask

By default, exclude initial state from loss:
```python
W = ones_like(X)
W[:, 0, :] = 0  # Don't penalize initial state mismatch
```

## JAX-Specific Considerations

### JIT Compilation

Critical functions are JIT-compiled for performance:
```python
@jit
def train_step(params, batch):
    loss, grads = value_and_grad(loss_fn)(params)
    ...
    return updated_params
```

### Gradient Flow

MuJoCo MJX provides analytical gradients through physics:
```python
# Forward pass (differentiable)
X1_hat = rollout(x0, U_hat)

# Backward pass (automatic)
grad_U = jax.grad(lambda U: loss(rollout(x0, U)))(U_hat)
```

### Batching

All operations are fully batched:
- States: (batch, H+1, state_dim)
- Actions: (batch, H, action_dim)
- MJX data: batched across all environments

## Performance Tips

### 1. Batch Size
- Larger batches → better GPU utilization
- Typical: 64-1024 depending on GPU memory

### 2. Horizon Length
- Longer horizons → more computation per step
- Default: H=16 (0.32s at 50Hz)

### 3. ODE Steps
- Single step (dt=1) sufficient for training
- Multi-step (dt=0.1) for higher quality sampling

### 4. Model Size
- Small (d_model=128, depth=2): ~1M params, fast
- Medium (d_model=256, depth=3): ~5M params, balanced
- Large (d_model=384, depth=3): ~10M params, highest quality

## Differences from PyTorch Implementation

| Aspect | PyTorch/Genesis | JAX/MuJoCo |
|--------|----------------|------------|
| Backend | Genesis | MuJoCo MJX |
| Auto-diff | PyTorch autograd | JAX grad |
| Acceleration | GPU via Genesis | GPU via MJX |
| Batching | Manual env replication | Native JAX vmap |
| Compilation | None | JIT compilation |
| Parallelism | Multi-env Genesis | Batched MJX |

## Validation

To verify your implementation matches the paper:

1. **Zero SAE**: Rollout should produce physically consistent trajectories
   ```python
   X1 = rollout(x0, U)
   # Check: all transitions are feasible
   ```

2. **Low TRE**: Generated trajectories should match data distribution
   ```python
   X_pred = sample(x0, params, rollout)
   TRE = mean((X_pred - X_true)²)
   # Target: TRE < 0.01 on feasible data
   ```

3. **Gradient flow**: Ensure gradients propagate through rollout
   ```python
   grad_fn = jax.grad(lambda p: loss_fn(p, batch))
   grads = grad_fn(params)
   # Check: grads are not NaN/Inf
   ```

## Troubleshooting

### Common Issues

1. **NaN gradients**: 
   - Check for division by zero in velocity_field
   - Ensure (1-t) is clamped away from 0

2. **Slow training**:
   - Ensure JIT compilation is enabled
   - Use larger batch sizes
   - Check GPU utilization

3. **Physics instability**:
   - Verify joint limits are enforced
   - Check timestep dt matches XML
   - Ensure initial states are valid

4. **Memory errors**:
   - Reduce batch size
   - Reduce horizon H
   - Use gradient accumulation

## References

- Paper: [DynaFlow (arXiv:2509.19804)](https://arxiv.org/abs/2509.19804)
- MuJoCo MJX: [https://mujoco.readthedocs.io/en/stable/mjx.html](https://mujoco.readthedocs.io/en/stable/mjx.html)
- JAX: [https://github.com/google/jax](https://github.com/google/jax)
- Flax: [https://github.com/google/flax](https://github.com/google/flax)
