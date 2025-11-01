"""
Collect trajectories from trained RL policy for DynaFlow training

This script loads a trained PPO policy and collects state-action trajectories
in the format required for training the DynaFlow model.
"""

import os
import argparse
import pickle
import torch
import numpy as np
from tqdm import tqdm

from env_wrapper import Go2MuJoCoEnv
from rsl_rl.runners import OnPolicyRunner


def extract_state_from_mujoco(env: Go2MuJoCoEnv, env_idx: int) -> np.ndarray:
    """
    Extract 37D state vector from MuJoCo data
    
    State format (37D):
    - base_pos (3): x, y, z
    - base_quat (4): w, x, y, z
    - joint_pos (12): 12 joint angles
    - base_lin_vel (3): linear velocity in base frame
    - base_ang_vel (3): angular velocity in base frame
    - joint_vel (12): 12 joint velocities
    """
    mj_data = env.mj_datas[env_idx]
    
    # Extract from qpos and qvel
    base_pos = mj_data.qpos[:3].copy()  # (3,)
    base_quat = mj_data.qpos[3:7].copy()  # (4,)
    joint_pos = mj_data.qpos[7:19].copy()  # (12,)
    
    # Velocities (world frame from MuJoCo)
    base_lin_vel_world = mj_data.qvel[:3].copy()  # (3,)
    base_ang_vel_world = mj_data.qvel[3:6].copy()  # (3,)
    joint_vel = mj_data.qvel[6:18].copy()  # (12,)
    
    # Transform velocities to base frame
    from env_wrapper import quat_rotate_inverse_np
    base_lin_vel = quat_rotate_inverse_np(base_quat, base_lin_vel_world)
    base_ang_vel = quat_rotate_inverse_np(base_quat, base_ang_vel_world)
    
    # Concatenate to 37D state
    state = np.concatenate([
        base_pos,      # 3
        base_quat,     # 4
        joint_pos,     # 12
        base_lin_vel,  # 3
        base_ang_vel,  # 3
        joint_vel,     # 12
    ])
    
    return state


def extract_conditioning_from_obs(env: Go2MuJoCoEnv, env_idx: int) -> np.ndarray:
    """
    Extract conditioning vector from environment state
    
    Conditioning format (38D):
    - commands (5): lin_vel_x, lin_vel_y, ang_vel_z, height, jump
    - base_lin_vel (3): current linear velocity
    - base_ang_vel (3): current angular velocity
    - joint_pos (12): current joint positions
    - joint_vel (12): current joint velocities
    - gravity (3): projected gravity vector
    """
    mj_data = env.mj_datas[env_idx]
    
    # Get unscaled values
    base_quat = mj_data.qpos[3:7].copy()
    base_lin_vel_world = mj_data.qvel[:3].copy()
    base_ang_vel_world = mj_data.qvel[3:6].copy()
    
    # Transform to base frame
    from env_wrapper import quat_rotate_inverse_np
    base_lin_vel = quat_rotate_inverse_np(base_quat, base_lin_vel_world)
    base_ang_vel = quat_rotate_inverse_np(base_quat, base_ang_vel_world)
    
    # Get joint states
    joint_pos = mj_data.qpos[7:19].copy()
    joint_vel = mj_data.qvel[6:18].copy()
    
    # Get projected gravity
    gravity_world = np.array([0., 0., -1.])
    projected_gravity = quat_rotate_inverse_np(base_quat, gravity_world)
    
    # Get commands (convert from torch to numpy)
    commands = env.commands[env_idx].cpu().numpy()
    
    # Concatenate conditioning (38D)
    conditioning = np.concatenate([
        commands,          # 5
        base_lin_vel,      # 3
        base_ang_vel,      # 3
        joint_pos,         # 12
        joint_vel,         # 12
        projected_gravity, # 3
    ])
    
    return conditioning


def collect_trajectories(
    env: Go2MuJoCoEnv,
    policy,
    num_episodes: int,
    max_steps: int,
    deterministic: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Collect trajectories from trained policy
    
    Args:
        env: Go2MuJoCoEnv instance
        policy: Trained policy (callable)
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
        deterministic: Use deterministic policy (not used for rsl_rl)
        verbose: Print progress
        
    Returns:
        data: Dictionary with trajectories and conditioning
    """
    all_trajectories = []
    all_conditioning = []
    all_actions = []
    
    episodes_collected = 0
    pbar = tqdm(total=num_episodes, desc="Collecting episodes") if verbose else None
    
    # Track episode data for each parallel environment
    episode_states = [[] for _ in range(env.num_envs)]
    episode_conditioning = [[] for _ in range(env.num_envs)]
    episode_actions = [[] for _ in range(env.num_envs)]
    
    # Reset environment
    obs, _ = env.reset()
    
    for step in range(max_steps):
        # Extract state and conditioning for each environment
        for env_idx in range(env.num_envs):
            state = extract_state_from_mujoco(env, env_idx)
            conditioning = extract_conditioning_from_obs(env, env_idx)
            
            episode_states[env_idx].append(state)
            episode_conditioning[env_idx].append(conditioning)
        
        # Sample action from policy
        with torch.no_grad():
            action = policy(obs)
        action_np = action.cpu().numpy()
        
        # Store actions
        for env_idx in range(env.num_envs):
            episode_actions[env_idx].append(action_np[env_idx])
        
        # Step environment
        obs, _, _, dones, _ = env.step(action)
        
        # Check for episode termination
        done_indices = torch.where(dones)[0].cpu().numpy()
        
        for idx in done_indices:
            if episodes_collected >= num_episodes:
                break
            
            # Extract trajectory for this environment
            traj_states = np.array(episode_states[idx])
            traj_cond = np.array(episode_conditioning[idx])
            traj_actions = np.array(episode_actions[idx])
            
            # Only save if trajectory is long enough
            if len(traj_states) >= 10:
                all_trajectories.append(traj_states)
                all_conditioning.append(traj_cond)
                all_actions.append(traj_actions)
                episodes_collected += 1
                
                if pbar is not None:
                    pbar.update(1)
            
            # Reset this environment's episode data
            episode_states[idx] = []
            episode_conditioning[idx] = []
            episode_actions[idx] = []
        
        # If we've collected enough episodes, break
        if episodes_collected >= num_episodes:
            break
    
    if pbar is not None:
        pbar.close()
    
    # Convert to object arrays (ragged arrays)
    trajectories_array = np.empty(len(all_trajectories), dtype=object)
    for i, traj in enumerate(all_trajectories):
        trajectories_array[i] = traj
    
    conditioning_array = np.empty(len(all_conditioning), dtype=object)
    for i, cond in enumerate(all_conditioning):
        conditioning_array[i] = cond
    
    actions_array = np.empty(len(all_actions), dtype=object)
    for i, act in enumerate(all_actions):
        actions_array[i] = act
    
    return {
        'trajectories': trajectories_array,
        'conditioning': conditioning_array,
        'actions': actions_array,
    }


def main():
    parser = argparse.ArgumentParser(description="Collect trajectories from trained RL policy")
    
    # Policy args
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained policy checkpoint (model_*.pt)")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config file (cfgs.pkl from training)")
    
    # Environment args
    parser.add_argument("--xml-path", type=str, default=None,
                       help="Path to MuJoCo XML model (optional, will use config)")
    parser.add_argument("--num-envs", type=int, default=64,
                       help="Number of parallel environments")
    
    # Collection args
    parser.add_argument("--num-episodes", type=int, default=100,
                       help="Number of episodes to collect")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximum steps per episode")
    
    # Output args
    parser.add_argument("--output", type=str, default="logs/trajectories_rl.npz",
                       help="Output file path")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for policy")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Trajectory Collection Configuration")
    print("="*60)
    for arg in vars(args):
        print(f"{arg:20s}: {getattr(args, arg)}")
    print("="*60)
    print()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'rb') as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    
    # Override num_envs if specified
    xml_path = args.xml_path
    
    # Create environment
    print(f"Creating environment with {args.num_envs} parallel environments...")
    env = Go2MuJoCoEnv(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        xml_path=xml_path,
    )
    
    # Create dummy runner to load policy
    print(f"Loading checkpoint from {args.checkpoint}...")
    # We need to create a temporary directory for the runner
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = OnPolicyRunner(env, train_cfg, temp_dir, device=args.device)
        runner.load(args.checkpoint, load_optimizer=False)
        policy = runner.get_inference_policy(device=args.device)
    
    print(f"Policy loaded successfully")
    
    # Collect trajectories
    print(f"\nCollecting {args.num_episodes} episodes...")
    data = collect_trajectories(
        env,
        policy,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        deterministic=False,
        verbose=True,
    )
    
    # Save data
    print(f"\nSaving trajectories to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    np.savez(
        args.output,
        trajectories=data['trajectories'],
        conditioning=data['conditioning'],
        actions=data['actions'],
    )
    
    # Print statistics
    print("\nCollection complete!")
    print(f"Number of episodes: {len(data['trajectories'])}")
    traj_lengths = [len(traj) for traj in data['trajectories']]
    print(f"Trajectory lengths: min={min(traj_lengths)}, max={max(traj_lengths)}, mean={np.mean(traj_lengths):.1f}")
    print(f"State dimension: {data['trajectories'][0].shape[-1]}")
    print(f"Conditioning dimension: {data['conditioning'][0].shape[-1]}")
    print(f"Action dimension: {data['actions'][0].shape[-1]}")
    print(f"\nData saved to {args.output}")


if __name__ == "__main__":
    main()


"""
Usage example:

# Collect trajectories from trained policy
python collect_trajectories.py \\
    --checkpoint logs/go2_ppo/model_1000.pt \\
    --config logs/go2_ppo/cfgs.pkl \\
    --num-episodes 100 \\
    --num-envs 64 \\
    --output logs/trajectories_rl.npz

# With custom XML path
python collect_trajectories.py \\
    --checkpoint logs/go2_ppo/model_1000.pt \\
    --config logs/go2_ppo/cfgs.pkl \\
    --xml-path Unitree_go2/custom_scene.xml \\
    --num-episodes 200 \\
    --output logs/trajectories_custom.npz
"""
