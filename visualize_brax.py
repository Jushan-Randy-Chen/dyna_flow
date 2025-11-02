#!/usr/bin/env python3
"""
Visualize trained Go2 policy using Brax/MJX.

Loads a trained policy and renders the robot's behavior.
"""

import argparse
import os

import jax
from jax import numpy as jp
import numpy as np
import mujoco
import mediapy as media

from brax import envs
from brax.io import model, html

from go2_env_brax import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Go2 policy")
    parser.add_argument("-e", "--exp_name", type=str, required=True,
                        help="Experiment name (must match training)")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of steps to simulate (default: 1000)")
    parser.add_argument("--render_every", type=int, default=2,
                        help="Render every N steps (default: 2)")
    parser.add_argument("--camera", type=str, default="track",
                        choices=["track", "side", "front", "top"],
                        help="Camera view (default: track)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (default: logs/{exp_name}/rollout.mp4)")
    parser.add_argument("--commands", type=float, nargs=3, default=None,
                        help="Manual commands [vx vy vyaw] (optional)")
    parser.add_argument("--html", action="store_true",
                        help="Also generate interactive HTML visualization")
    args = parser.parse_args()
    
    # Paths
    log_dir = f"logs/{args.exp_name}"
    model_path = f"{log_dir}/policy"
    
    if args.output is None:
        args.output = f"{log_dir}/rollout.mp4"
    
    print("=" * 80)
    print("Go2 Policy Visualization")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Model path: {model_path}")
    print(f"Steps: {args.n_steps}")
    print(f"Camera: {args.camera}")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print(f"   Make sure you've trained a model first:")
        print(f"   python train_brax.py --exp_name {args.exp_name}")
        return
    
    print("\nLoading policy...")
    params = model.load_params(model_path)
    print("✓ Policy loaded")
    
    # Register and create environment
    envs.register_environment('go2', Go2Env)
    env = envs.get_environment('go2')
    print(f"✓ Environment created")
    
    # Create inference function (this comes from training, we need to rebuild it)
    from brax.training.agents.ppo import networks as ppo_networks
    import functools
    
    make_networks = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128),
    )
    
    network = make_networks(
        observation_size=env.observation_size,
        action_size=env.action_size,
        preprocess_observations_fn=lambda x, rng=None: x,
    )
    
    make_policy = ppo_networks.make_inference_fn(network)
    inference_fn = make_policy(params)
    jit_inference_fn = jax.jit(inference_fn)
    
    print("✓ Inference function created")
    
    # Reset environment
    print(f"\nRunning rollout (seed={args.seed})...")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    
    # Override commands if provided
    if args.commands:
        print(f"Using manual commands: vx={args.commands[0]}, vy={args.commands[1]}, "
              f"vyaw={args.commands[2]}")
        state.info['command'] = jp.array(args.commands)
    else:
        print(f"Using random commands: {state.info['command']}")
    
    rollout = [state.pipeline_state]
    
    # Debug initial state
    torso_height = state.pipeline_state.x.pos[0, 2]  # Assuming torso is at index 0
    print(f"\nInitial state diagnostics:")
    print(f"  Torso height: {torso_height:.4f} m")
    print(f"  Done flag: {state.done}")
    print(f"  Observation shape: {state.obs.shape}")
    print(f"  Observation mean: {jp.mean(state.obs):.4f}, std: {jp.std(state.obs):.4f}")
    print(f"  Observation min/max: {jp.min(state.obs):.4f} / {jp.max(state.obs):.4f}")
    
    # Test first action
    act_rng_test, _ = jax.random.split(rng)
    ctrl_test, _ = jit_inference_fn(state.obs, act_rng_test)
    print(f"  First action mean: {jp.mean(ctrl_test):.4f}, std: {jp.std(ctrl_test):.4f}")
    print(f"  First action min/max: {jp.min(ctrl_test):.4f} / {jp.max(ctrl_test):.4f}")
    
    # Collect trajectory
    print(f"\nStarting rollout...")
    for i in range(args.n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        
        # Debug first few steps
        if i < 3:
            print(f"  Step {i}: action mean={jp.mean(ctrl):.4f}, "
                  f"height={state.pipeline_state.x.pos[0, 2]:.4f}, done={state.done}")
        
        if state.done:
            torso_height = state.pipeline_state.x.pos[0, 2]
            up = jp.array([0.0, 0.0, 1.0])
            from brax import math as brax_math
            upright = jp.dot(brax_math.rotate(up, state.pipeline_state.x.rot[0]), up)
            
            # Check joint angles
            joint_angles = state.pipeline_state.q[7:]  # Skip base pose
            lowers = np.array([-0.7, -1.0, -2.2] * 4) # lower bound on the joint angles
            uppers = np.array([0.52, 2.1, -0.4] * 4) # upper bound on the joint angles
            below_lower = joint_angles < lowers
            above_upper = joint_angles > uppers
            
            print(f"\n  ❌ Episode terminated at step {i}")
            print(f"  Termination diagnostics:")
            print(f"    - Torso height: {torso_height:.4f} m (threshold: 0.18)")
            print(f"    - Upright dot product: {upright:.4f} (threshold: 0.0)")
            print(f"    - Joint angles below lower limit: {jp.any(below_lower)} {jp.where(below_lower)[0]}")
            print(f"    - Joint angles above upper limit: {jp.any(above_upper)} {jp.where(above_upper)[0]}")
            if jp.any(below_lower) or jp.any(above_upper):
                print(f"    - Problematic joint angles:")
                for idx in jp.where(below_lower | above_upper)[0]:
                    print(f"      Joint {int(idx)}: {joint_angles[int(idx)]:.4f} "
                          f"(limits: [{lowers[int(idx)]:.2f}, {uppers[int(idx)]:.2f}])")
            break
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i+1}/{args.n_steps}")
    
    print(f"✓ Rollout complete ({len(rollout)} states)")
    
    # Render video
    print(f"\nRendering video...")
    frames = env.render(rollout[::args.render_every], camera=args.camera)
    fps = 1.0 / env.dt / args.render_every
    
    # Save video
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    media.write_video(args.output, frames, fps=fps)
    print(f"✓ Video saved to {args.output}")
    
    # Generate HTML if requested
    if args.html:
        html_path = args.output.replace('.mp4', '.html')
        print(f"\nGenerating interactive HTML...")
        html_string = html.render(
            env.sys.tree_replace({'opt.timestep': env.dt}),
            rollout[::args.render_every]
        )
        with open(html_path, 'w') as f:
            f.write(html_string)
        print(f"✓ HTML saved to {html_path}")
        print(f"  Open in browser to interact with 3D visualization")
    
    print("\n" + "=" * 80)
    print("Visualization complete! 🎥")
    print("=" * 80)
    print(f"\nWatch the video: {args.output}")
    if args.html:
        print(f"Interactive 3D: {html_path}")


if __name__ == "__main__":
    main()


"""
Usage Examples:

# Basic visualization of trained policy:
python visualize_brax.py --exp_name ppo_policy

# Longer rollout with different camera:
python visualize_brax.py --exp_name ppo_policy --n_steps 2000 --camera side

# Manual command (walk forward at 1 m/s):
python visualize_brax.py --exp_name ppo_policy --commands 1.0 0.0 0.0 0.3 0.0

# Manual command (turn right while walking):
python visualize_brax.py --exp_name ppo_policy --commands 0.5 0.0 0.5 0.3 0.0

# Generate interactive HTML visualization:
python visualize_brax.py --exp_name ppo_policy --html

# Custom output path:
python visualize_brax.py --exp_name ppo_policy --output my_rollout.mp4

Available cameras:
- track: Follows the robot (default)
- side: Side view
- front: Front view
- top: Top-down view
"""
