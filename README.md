# Implementation of the flow-matching policy proposed in Dynamics-embedded Flow Matching for Physically Consistent Motion Generation

**Dynamics-embedded Flow Matching for Physically Consistent Motion Generation**

[![Paper](https://img.shields.io/badge/arXiv-2509.19804-b31b1b.svg)](https://arxiv.org/html/2509.19804v2)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This is a JAX/MuJoCo implementation of **DynaFlow**, which embeds a differentiable physics simulator directly into a flow matching model to guarantee physically consistent motion generation from state-only demonstrations.

**Ackowledgement & disclaimer: I am not the author of this work!! This is my own implementation following the paper. 

## 🎯 Overview

DynaFlow generates trajectories in the **action space** and maps them to dynamically feasible **state trajectories** via a differentiable simulator. This ensures all outputs are physically consistent by construction.

**Key Innovation:** Instead of learning state transitions directly, DynaFlow learns to predict actions U, then uses a differentiable physics simulator Φ to compute the resulting state trajectory:

```
X̂₁ = Φ(x₀, U)  where  U = D_θ(X_t, c, t)
```

### Technology Stack

- **JAX** - Automatic differentiation and XLA compilation
- **MuJoCo MJX** - GPU-accelerated differentiable physics
- **Flax/Linen** - Neural network library
- **Optax** - Gradient-based optimization

### Key Features

✅ Differentiable physics rollout: Φ(x₀, U) → X  
✅ 1D Diffusion Transformer (DiT) for action prediction  
✅ Conditional flow matching with state-only demonstrations  
✅ Single-step ODE sampling for real-time inference  
✅ Continuous conditioning (observations, commands, gait modes)  
✅ PPO trajectory collection pipeline

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- MuJoCo 3.0+

### Quick Install

```bash
cd dyna_flow
pip install -e .
```

This installs:
- `jax[cuda]` - GPU acceleration
- `mujoco>=3.0.0` and `mujoco-mjx>=3.0.0` - Physics simulation
- `flax` and `optax` - Neural networks and optimization
- `numpy`, `pandas` - Data handling

### Optional Dependencies

```bash
pip install wandb tqdm  # For experiment tracking and progress bars
```

### Verify Installation

```bash
python -c "import jax; import mujoco; from mujoco import mjx; print('✓ Ready to go!')"
```

---

## 🚀 Quick Start

### 1. Collect Demonstration Data

**Option A: Train PPO Policy** (Recommended)

Train a PPO policy to collect high-quality demonstrations:

```bash
# Train policy (1-2 hours on GPU)
python train_ppo.py \
  --exp_name go2-locomotion \
  --num_envs 4096 \
  --max_iterations 10000

# Collect trajectories from trained policy
python collect_trajectories_parallel.py \
  --checkpoint logs/go2-locomotion/model_10000.pt \
  --num-episodes 10000 \
  --output logs/go2-locomotion/trajectories/trajectories.npz
```

**Option B: Use Existing Data**

Format your data as NPZ with the following structure:

```python
import numpy as np

# Save trajectories
np.savez(
    "my_data.npz",
    states=np.array(states_list),        # (N, T, 37) - state trajectories
    conditionings=np.array(conds_list),  # (N, T, cond_dim) - conditioning vectors (optional)
    actions=np.array(actions_list)       # (N, T, 12) - actions (optional)
)
```

**State dimension (37):** `[base_pos(3), base_quat(4), joint_pos(12), base_vel(3), base_angvel(3), joint_vel(12)]`

---

## 🎓 Training

### Full Training Configuration

```
python train_flow_matching.py \
    --data logs/ppo_policy2/trajectories/trajectories.npz \
    --wandb online \
    --wandb-project dynaflow-go2

```
(you can disable wandb by ``--wandb disabled`)
### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data` | Path to NPZ dataset | Required |
| `--epochs` | Number of training epochs | 100 |
| `--batch` | Batch size | 32 |
| `--horizon` | Trajectory prediction horizon | 10 |
| `--lr` | Learning rate | 1e-4 |
| `--weight-decay` | AdamW weight decay | 0.0 |
| `--ema-decay` | Exponential moving average decay | 0.995 |
| `--d-model` | Transformer hidden dimension | 384 |
| `--n-heads` | Number of attention heads | 6 |
| `--depth` | Number of transformer blocks | 3 |
| `--dropout` | Dropout rate | 0.1 |
| `--save-dir` | Checkpoint save directory | `checkpoints` |
| `--save-interval` | Save every N epochs | 10 |

<!-- ### Monitoring Training

If using Weights & Biases:

```bash
python train_flow_matching.py \
  --data my_data.npz \
  --wandb online \
  --wandb-project my-project \
  --wandb-name my-experiment
``` -->

<!-- Visit https://wandb.ai to monitor:
- Training loss curves
- Learning rate schedule
- Gradient norms
- Sample quality metrics -->

---

<!-- ## 📊 Evaluation & Sampling

### Evaluate on Test Set

```bash
python evaluate.py \
  --checkpoint checkpoints/dynaflow/epoch_200.pkl \
  --dataset test_data.npz \
  --num-samples 100 \
  --batch 32 \
  --ode-steps 1 \
  --use-ema
``` -->

## 🏗️ Architecture

### Model Components

**1. Action Predictor (D_θ)**
- **Type:** 1D Diffusion Transformer (DiT)
- **Input:** Noisy state trajectory X_t, diffusion time t, conditioning c
- **Output:** Predicted action sequence U
- **Architecture:**
  - Sinusoidal position embeddings
  - Time embedding with Mish activation
  - Continuous conditioning via attention-based embedder
  - Multi-head self-attention blocks with adaLN-Zero
  - Final projection to action space

**2. Differentiable Rollout (Φ)**
- **Type:** MuJoCo MJX physics simulator
- **Input:** Initial state x₀, action sequence U
- **Output:** Full state trajectory X
- **Features:**
  - Batched parallel simulation via `jax.vmap`
  - Fully differentiable for gradient-based training
  - GPU-accelerated for efficiency

**3. Flow Matching Loss**

Conditional Matching Loss (Equation 4 from paper):

```
L(θ) = E[||W ⊙ (X̂₁ - X₁)||²]
```

where:
- `X̂₁ = Φ(x₀, D_θ(X_t, c, t))` - Predicted trajectory via simulator
- `X₁` - Ground truth trajectory from demonstrations
- `W` - Weighting matrix (optional, for joint-specific weights)
- `X_t = t·X₁ + (1-t)·X₀` - Interpolated noisy trajectory
- `t ~ U(0,1)` - Diffusion time

### Training Pipeline

```
1. Sample batch of demonstrations: (x₀, X₁, c)
2. Sample diffusion time: t ~ U(0,1)
3. Create noisy trajectory: X_t = t·X₁ + (1-t)·X₀
4. Predict actions: U = D_θ(X_t, c, t)
5. Simulate trajectory: X̂₁ = Φ(x₀, U)
6. Compute loss: L = ||X̂₁ - X₁||²
7. Update parameters: θ ← θ - ∇L
```

### Inference Pipeline

```
1. Start with noise: X₀ ~ N(X₀|x₀, σ²I)
2. For t from 1 to 0 (single-step ODE):
   a. Predict actions: U = D_θ(X_t, c, t)
   b. Simulate: X̂₁ = Φ(x₀, U)
   c. Update: X_t ← X̂₁
3. Return final trajectory: X̂₁
```

---

## 📁 Project Structure

```
dyna_flow/
├── model.py                    # DiT transformer architecture
├── rollout.py                  # MuJoCo MJX differentiable simulator
├── losses.py                   # Flow matching loss and sampling
├── data.py                     # Dataset loading utilities
├── utils.py                    # Helper functions
├── train_flow_matching.py      # Main training script
├── evaluate.py                 # Evaluation script
├── train_ppo.py                # PPO policy training
├── collect_trajectories_parallel.py  # Trajectory collection
├── Unitree_go2/                # MuJoCo XML models
│   └── scene_mjx_gym.xml
└── README.md                   # This file
```

---

<!-- ## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{lee2025dynaflow,
  title={DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent 
         Motion Generation from State-only Demonstrations},
  author={Lee, Sowoo and Kang, Dongyun and Park, Jaehyun and Park, Hae-Won},
  journal={arXiv preprint arXiv:2509.19804},
  year={2025}
}
``` -->

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Original DynaFlow paper and authors at Korea Advanced Institute of Science & Technology (KAIST)

