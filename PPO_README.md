# PPO Training for DynaFlow

This directory contains a PPO (Proximal Policy Optimization) implementation for training the Unitree Go2 robot using the `rsl_rl` library. The implementation uses the exact same parameters and configuration as `go2_train.py` from the quadrupeds_locomotion project.

## Files

- **`train_ppo.py`**: Main training script with PPO algorithm
- **`env_wrapper.py`**: MuJoCo environment wrapper implementing the `VecEnv` interface required by rsl_rl

## Configuration

The implementation uses identical parameters to `go2_train.py`:

### PPO Algorithm Parameters
- **clip_param**: 0.2
- **desired_kl**: 0.01
- **entropy_coef**: 0.01
- **gamma**: 0.99
- **lam**: 0.95
- **learning_rate**: 0.001
- **max_grad_norm**: 1.0
- **num_learning_epochs**: 5
- **num_mini_batches**: 4
- **schedule**: adaptive
- **use_clipped_value_loss**: True
- **value_loss_coef**: 1.0

### Policy Network Architecture
- **Actor hidden dims**: [512, 256, 128]
- **Critic hidden dims**: [512, 256, 128]
- **Activation**: ELU
- **Init noise std**: 1.0

### Training Configuration
- **num_steps_per_env**: 24
- **save_interval**: 100 iterations
- **log_interval**: 1 iteration

## Installation

### Prerequisites

1. Install rsl_rl:
```bash
cd ../quadrupeds_locomotion/rsl_rl
pip install -e .
```

2. Ensure MuJoCo dependencies are installed:
```bash
pip install mujoco>=3.0.0
```

3. Install PyTorch (if not already installed):
```bash
pip install torch
```

## Usage

### Basic Training

```bash
python train_ppo.py
```

This will:
- Create 4096 parallel environments
- Train for 10,000 iterations
- Save checkpoints every 100 iterations to `logs/go2-ppo-dynaflow/`

### Custom Configuration

```bash
# Change number of environments
python train_ppo.py --num_envs 2048

# Change experiment name
python train_ppo.py --exp_name my_experiment

# Adjust training iterations
python train_ppo.py --max_iterations 5000

# Use CPU instead of GPU
python train_ppo.py --device cpu --num_envs 512

# Specify custom XML path
python train_ppo.py --xml-path /path/to/custom/go2.xml
```

### Full Example

```bash
python train_ppo.py \
    --exp_name go2_custom \
    --num_envs 2048 \
    --max_iterations 8000 \
    --device cuda:0
```

## Environment Details

### Observation Space (48D)
- Linear velocity (3D)
- Angular velocity (3D)
- Gravity vector in base frame (3D)
- Commands (5D): [lin_vel_x, lin_vel_y, ang_vel, height, jump]
- Joint positions relative to default (12D)
- Joint velocities (12D)
- Last actions (12D)

### Action Space (12D)
PD position targets for the 12 joints:
- Front Right: hip, thigh, calf
- Front Left: hip, thigh, calf
- Rear Right: hip, thigh, calf
- Rear Left: hip, thigh, calf

Actions are residuals from the default pose, scaled by `action_scale=0.3`

### Reward Function

The reward is a weighted sum of the following components:

**Tracking rewards:**
- Linear velocity tracking (weight: 1.0)
- Angular velocity tracking (weight: 0.2)

**Penalties:**
- Vertical velocity (weight: -1.0)
- Base height deviation (weight: -50.0)
- Action rate (weight: -0.005)
- Deviation from default pose (weight: -0.1)

### Termination Conditions
- Roll angle > 10 degrees
- Pitch angle > 10 degrees
- Episode length exceeds 20 seconds

## Training Outputs

Checkpoints are saved to `logs/{experiment_name}/`:
- `model_{iteration}.pt`: Model checkpoints
- `cfgs.pkl`: Configuration pickle file
- TensorBoard logs for training metrics

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

## Comparison with go2_train.py

This implementation is functionally equivalent to `go2_train.py` but uses:
- **MuJoCo** instead of Genesis for physics simulation
- **PyTorch** instead of JAX
- **CPU/GPU** flexibility (Genesis uses GPU by default)

All training hyperparameters, network architectures, and reward functions are identical.

## Collecting Demonstration Data

After training, you can use the learned policy to collect demonstration trajectories for DynaFlow:

```python
import torch
import pickle

# Load trained model
model_path = "logs/go2-ppo-dynaflow/model_10000.pt"
checkpoint = torch.load(model_path)

# Use policy for data collection
# (implementation to be added in collect_trajectories.py)
```

## Troubleshooting

### ImportError: No module named 'rsl_rl'

Install rsl_rl:
```bash
cd ../quadrupeds_locomotion/rsl_rl
pip install -e .
```

Or add to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/quadrupeds_locomotion/rsl_rl
```

### MuJoCo XML not found

Specify the XML path explicitly:
```bash
python train_ppo.py --xml-path Unitree_go2/go2_mjx_gym.xml
```

### CUDA out of memory

Reduce the number of environments:
```bash
python train_ppo.py --num_envs 1024
```

Or use CPU:
```bash
python train_ppo.py --device cpu --num_envs 512
```

## References

- **rsl_rl**: https://github.com/leggedrobotics/rsl_rl
- **PPO Paper**: https://arxiv.org/abs/1707.06347
- **Original go2_train.py**: `../quadrupeds_locomotion/src/go2_train.py`
