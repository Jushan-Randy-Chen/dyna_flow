# DynaFlow - JAX/MuJoCo Implementation

This is a JAX/MuJoCo reimplementation of **DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations**.

Paper: https://arxiv.org/html/2509.19804v2

## Overview

DynaFlow embeds a differentiable simulator directly into a flow matching model to guarantee physically consistent motion generation. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction.

This implementation uses:
- **JAX** for automatic differentiation and high-performance computation
- **MuJoCo MJX** for massively parallel differentiable physics simulation
- **Flax** for neural network implementation (DiT transformer)
- **Optax** for optimization

## Key Features

- ✅ Differentiable MuJoCo rollout operator Φ: (x₀, U) → X₁
- ✅ 1D Diffusion Transformer (DiT) for action prediction
- ✅ Conditional flow matching loss with state-only demonstrations
- ✅ Single-step ODE integration for real-time inference
- ✅ Support for conditioning on observations, commands, and gait modes
- ✅ **Complete RL training pipeline** for collecting demonstration data (PPO in MuJoCo)

## Installation

### Option 1: Using pip
```bash
# Install from source
cd dyna_flow
pip install -e .
```

### Option 2: Manual installation
```bash
pip install jax jaxlib mujoco>=3.0.0 mujoco-mjx>=3.0.0 flax optax numpy pandas
pip install wandb tqdm  # optional, for logging and progress bars
```

### Verify Installation
```bash
python -c "import jax; import mujoco; from mujoco import mjx; print('✓ All dependencies installed')"
```

## Quick Start

### 1. Run the Example
```bash
cd dyna_flow
python example.py
```

This will train a small model on synthetic data and demonstrate sampling.

### 2. Generate Training Data (PPO RL Pipeline)

**NEW**: Train a PPO policy using rsl_rl and collect demonstration trajectories:

```bash
# Step 1: Train PPO policy (takes ~1-2 hours on GPU)
python train_ppo.py --exp_name go2-ppo --num_envs 4096 --max_iterations 10000

# Step 2: Collect trajectories from trained policy (implementation pending)
# python collect_trajectories.py \
#   --checkpoint logs/go2-ppo/model_10000.pt \
#   --num-episodes 100 \
#   --output data/trajectories_ppo.npz
```

See [PPO_README.md](PPO_README.md) for detailed PPO training documentation.

### 3. Prepare Your Own Data (Alternative)

If you have existing trajectory data, format it as NPZ with shape `(N, T, state_dim)` where:
- `N` = number of episodes
- `T` = trajectory length
- `state_dim` = 37 for Go2 (base pos(3) + quat(4) + joints(12) + velocities(18))

Example data preparation:
```python
import numpy as np

# Your trajectories (list of variable-length episodes)
trajectories = [...]  # List of (T_i, 37) arrays

# Save as NPZ
np.savez("my_data.npz", trajectories=np.array(trajectories, dtype=object))

# Optional: include conditioning vectors
conditioning = [...]  # List of (T_i, cond_dim) arrays
np.savez("my_data.npz", 
         trajectories=np.array(trajectories, dtype=object),
         conds=np.array(conditioning, dtype=object))
```

### 4. Train DynaFlow Model

Basic training (XML path auto-detected from local `Unitree_go2/` folder):
```bash
python train.py \
  --data data/trajectories_rl.npz \
  --horizon 16 \
  --batch 64 \
  --epochs 100 \
  --save checkpoints/my_model.pkl
```

With full options:
```bash
python train.py \
  --data data/trajectories_rl.npz \
  --horizon 16 \
  --batch 1024 \
  --epochs 100 \
  --lr 2e-4 \
  --weight-decay 1e-4 \
  --ema-decay 0.995 \
  --d-model 384 \
  --n-heads 6 \
  --depth 3 \
  --save checkpoints/dynaflow.pkl \
  --save-interval 10 \
  --wandb online \
  --wandb-project dynaflow-experiments
```

> **Note:** The `--xml-path` argument is optional. By default, scripts will use `Unitree_go2/scene_mjx_gym.xml` from the local directory.

### 5. Evaluate the Model

```bash
python sample.py \
  --ckpt checkpoints/my_model.pkl \
  --dataset path/to/test_data.npz \
  --eval-samples 32 \
  --batch 8 \
  --ode-steps 1 \
  --use-ema \
  --save-samples results/samples.npz
```

### 5. Sample New Trajectories

```python
import jax
from jax import random
import jax.numpy as jnp
from dyna_flow.utils import load_checkpoint
from dyna_flow.model import ActionPredictor
from dyna_flow.rollout import create_go2_rollout
from dyna_flow.losses import sample_trajectory

# Load checkpoint
ckpt = load_checkpoint("checkpoints/my_model.pkl")
params = ckpt['ema_params']  # or ckpt['params']

# Create model and rollout
model = ActionPredictor(
    state_dim=ckpt['state_dim'],
    action_dim=ckpt['action_dim'],
    cond_dim=ckpt.get('cond_dim')
)
rollout = create_go2_rollout(xml_path="path/to/scene_mjx_gym.xml")

# Sample trajectories
rng = random.PRNGKey(0)
x0 = jnp.array([[0, 0, 0.27, 1, 0, 0, 0, ...]])  # Initial state
X, U = sample_trajectory(
    model.apply, params, rollout, x0,
    horizon=16, state_dim=37, ode_steps=1, rng=rng
)

print(f"Sampled trajectory: {X.shape}")  # (1, 17, 37)
print(f"Actions: {U.shape}")  # (1, 16, 12)
```

## Architecture

### Action Predictor (DiT1d)
- 1D Diffusion Transformer with 3 blocks (configurable)
- Inputs: noisy state trajectory X_t, time t, optional conditioning c
- Outputs: action sequence U^

### Differentiable Rollout
- MuJoCo MJX-based simulator
- Maps (x₀, U) to full state trajectory X₁ through recursive dynamics
- Fully differentiable for end-to-end training

### Loss Function
Conditional Matching Loss (Eq. 4 from paper):
```
L(θ) = E[||W ⊙ (X̂₁ - X₁)||²]
```
where X̂₁ = Φ(x₀, D_θ(X_t, c, t))

## File Structure

```
dyna_flow/
├── __init__.py         # Package initialization
├── model.py            # JAX/Flax DiT transformer
├── rollout.py          # MuJoCo MJX differentiable rollout
├── losses.py           # Flow matching loss functions
├── data.py             # Dataset loading and preprocessing
├── utils.py            # Helper functions
├── train.py            # Training script
├── sample.py           # Sampling/evaluation script
└── README.md           # This file
```

## Citation

```bibtex
@article{lee2025dynaflow,
  title={DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations},
  author={Lee, Sowoo and Kang, Dongyun and Park, Jaehyun and Park, Hae-Won},
  journal={arXiv preprint arXiv:2509.19804},
  year={2025}
}
```
