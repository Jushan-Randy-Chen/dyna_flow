#!/usr/bin/env python3
"""
Inspect and visualize collected expert trajectories.

This script loads the trajectories.npz file and provides statistics
and visualizations of the collected expert demonstrations.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

# Import for HTML visualization
try:
    import jax
    from brax import envs
    from brax.base import Motion, State as BraxState, Transform
    from brax.io import html
    from go2_env_brax import Go2Env
    import jax.numpy as jp
    BRAX_AVAILABLE = True
except ImportError:
    BRAX_AVAILABLE = False
    print("Warning: Brax not available, HTML visualization will be disabled")

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description="Inspect collected trajectories")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to trajectories.npz file")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots")
    parser.add_argument("--html", action="store_true",
                        help="Generate interactive HTML visualization of first episode")
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Episode index to visualize in detail")
    parser.add_argument("--downsample", type=int, default=5,
                        help="HTML frame stride (higher is faster, default=5)")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"❌ Error: File not found: {args.data_path}")
        return
    
    print("=" * 80)
    print("Trajectory Data Inspection")
    print("=" * 80)
    print(f"Loading: {args.data_path}\n")
    
    # Load data
    data = np.load(args.data_path, allow_pickle=True)
    
    states = data['states']
    conditionings = data['conditionings']
    actions = data['actions']
    lengths = data['episode_lengths']
    
    n_episodes = len(states)
    
    print(f"Dataset Statistics:")
    print(f"  Number of episodes: {n_episodes}")
    print(f"  Total timesteps: {np.sum(lengths)}")
    print(f"  Episode length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"  Min/Max length: {np.min(lengths)} / {np.max(lengths)}")
    
    # Verify shapes
    print(f"\nData Dimensions:")
    print(f"  State dimension: {states[0].shape[1]} (expected 37)")
    print(f"  Conditioning dimension: {conditionings[0].shape[1]} (expected 36)")
    print(f"  Action dimension: {actions[0].shape[1]} (expected 12)")
    
    # Break down state components
    print(f"\nState Vector Breakdown (37D):")
    print(f"  [0:3]    Base position (x, y, z)")
    print(f"  [3:7]    Base orientation (quaternion w, x, y, z)")
    print(f"  [7:19]   Joint positions (12 joints)")
    print(f"  [19:22]  Base linear velocity")
    print(f"  [22:25]  Base angular velocity")
    print(f"  [25:37]  Joint velocities (12 joints)")
    
    print(f"\nConditioning Vector Breakdown (36D):")
    print(f"  [0:3]    Projected gravity (g)")
    print(f"  [3:6]    Base linear velocity in local frame (v)")
    print(f"  [6:9]    Base angular velocity in local frame (ω)")
    print(f"  [9:21]   Joint positions (q)")
    print(f"  [21:33]  Joint velocities (q̇)")
    print(f"  [33:36]  Velocity commands [vcmd_x, vcmd_y, ωcmd_z]")
    
    # Analyze commands
    all_commands = np.array([cond[:, 33:36] for cond in conditionings], dtype=object)
    print(f"\nCommand Distribution:")
    for i, name in enumerate(['vcmd_x', 'vcmd_y', 'ωcmd_z']):
        cmd_vals = np.concatenate([ep[:, i] for ep in all_commands])
        print(f"  {name:10s}: mean={np.mean(cmd_vals):6.3f}, "
              f"std={np.std(cmd_vals):6.3f}, "
              f"range=[{np.min(cmd_vals):6.3f}, {np.max(cmd_vals):6.3f}]")
    
    # Detailed episode inspection
    if args.episode_idx < n_episodes:
        print(f"\nEpisode {args.episode_idx} Details:")
        ep_state = np.array(states[args.episode_idx], dtype=np.float32)
        ep_cond = np.array(conditionings[args.episode_idx], dtype=np.float32)
        ep_action = np.array(actions[args.episode_idx], dtype=np.float32)
        
        print(f"  Length: {len(ep_state)} steps")
        print(f"  Command: vx={ep_cond[0, 33]:.2f}, vy={ep_cond[0, 34]:.2f}, "
              f"ω_z={ep_cond[0, 35]:.2f}")
        
        # Check for NaN or Inf
        has_nan_state = np.any(np.isnan(ep_state))
        has_inf_state = np.any(np.isinf(ep_state))
        has_nan_cond = np.any(np.isnan(ep_cond))
        has_inf_cond = np.any(np.isinf(ep_cond))
        has_nan_action = np.any(np.isnan(ep_action))
        has_inf_action = np.any(np.isinf(ep_action))
        
        print(f"  Data validity:")
        print(f"    States: NaN={has_nan_state}, Inf={has_inf_state}")
        print(f"    Conditionings: NaN={has_nan_cond}, Inf={has_inf_cond}")
        print(f"    Actions: NaN={has_nan_action}, Inf={has_inf_action}")
    
    # Generate plots
    if args.plot:
        print(f"\nGenerating plots...")
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Episode length histogram
        axes[0, 0].hist(lengths, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Command distribution
        all_cmds = np.array([np.array(cond[0, 33:36], dtype=np.float32) for cond in conditionings])
        axes[0, 1].scatter(all_cmds[:, 0], all_cmds[:, 1], alpha=0.5)
        axes[0, 1].set_xlabel('vcmd_x (m/s)')
        axes[0, 1].set_ylabel('vcmd_y (m/s)')
        axes[0, 1].set_title('Linear Velocity Commands')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Example episode - base trajectory
        ep_state = np.array(states[args.episode_idx], dtype=np.float32)
        ep_cond = np.array(conditionings[args.episode_idx], dtype=np.float32)
        
        axes[1, 0].plot(ep_state[:, 0], ep_state[:, 1])
        axes[1, 0].set_xlabel('X Position (m)')
        axes[1, 0].set_ylabel('Y Position (m)')
        axes[1, 0].set_title(f'Episode {args.episode_idx} - Base Trajectory')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        
        # Base height
        time = np.arange(len(ep_state)) * 0.02  # dt = 0.02
        axes[1, 1].plot(time, ep_state[:, 2])
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Height (m)')
        axes[1, 1].set_title(f'Episode {args.episode_idx} - Base Height')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Joint positions
        axes[2, 0].plot(time, ep_state[:, 7:19])
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Joint Angle (rad)')
        axes[2, 0].set_title(f'Episode {args.episode_idx} - Joint Positions')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Velocities
        axes[2, 1].plot(time, ep_cond[:, 3:6], label=['vx', 'vy', 'vz'])
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Velocity (m/s)')
        axes[2, 1].set_title(f'Episode {args.episode_idx} - Base Velocity')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(args.data_path), 'trajectory_analysis.png')
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Plot saved to {plot_path}")
        
        plt.show()
    
    # Generate HTML visualization
    if args.html:
        if not BRAX_AVAILABLE:
            print("\n❌ HTML visualization requires Brax. Install it to enable this feature.")
        else:
            import time

            print(f"\nGenerating HTML visualization for episode {args.episode_idx}...")

            # Register and create environment
            envs.register_environment('go2', Go2Env)
            env = envs.get_environment('go2')

            # Reconstruct pipeline states from the collected state data
            episode_raw = np.array(states[args.episode_idx], dtype=np.float32)

            # Downsample frames to control runtime
            downsample_factor = max(1, args.downsample)
            if downsample_factor > 1:
                downsampled = episode_raw[::downsample_factor]
                print(
                    f"  Downsampled to {len(downsampled)} frames (factor: {downsample_factor}×, "
                    f"original: {len(episode_raw)})"
                )
            else:
                downsampled = episode_raw
                print(f"  Using all {len(downsampled)} frames")

            # Slice state vector into generalized coordinates
            q_np = np.concatenate(
                [downsampled[:, 0:3], downsampled[:, 3:7], downsampled[:, 7:19]],
                axis=1,
            )
            qd_np = np.concatenate(
                [downsampled[:, 19:22], downsampled[:, 22:25], downsampled[:, 25:37]],
                axis=1,
            )

            q_all = jp.array(q_np)
            qd_all = jp.array(qd_np)

            use_mujoco = MUJOCO_AVAILABLE and getattr(env.sys, 'mj_model', None) is not None
            pipeline_states = []

            if use_mujoco:
                print("  Using MuJoCo forward kinematics for fast rendering")
                model = env.sys.mj_model
                data = mujoco.MjData(model)

                # Seed contact structure and link count from a single Brax state
                template_state = env.pipeline_init(q_all[0], qd_all[0])
                link_count = int(template_state.x.pos.shape[0])

                body_offset = model.nbody - link_count
                if body_offset not in (0, 1):
                    print("  ⚠️ Body count mismatch between Brax and MuJoCo; falling back to Brax pipeline")
                    use_mujoco = False
                else:
                    body_start = 0 if body_offset == 0 else 1
                    zeros_ang = jp.zeros((link_count, 3))
                    zeros_vel = jp.zeros((link_count, 3))

                    for i in range(len(downsampled)):
                        data.qpos[:] = q_np[i]
                        data.qvel[:] = qd_np[i]
                        mujoco.mj_forward(model, data)

                        pos = jp.array(
                            np.asarray(
                                data.xpos[body_start:body_start + link_count],
                                dtype=np.float32,
                            )
                        )
                        rot = jp.array(
                            np.asarray(
                                data.xquat[body_start:body_start + link_count],
                                dtype=np.float32,
                            )
                        )

                        x_tf = Transform(pos=pos, rot=rot)
                        xd_motion = Motion(ang=zeros_ang, vel=zeros_vel)

                        state = BraxState(
                            q=jp.array(q_np[i]),
                            qd=jp.array(qd_np[i]),
                            x=x_tf,
                            xd=xd_motion,
                            contact=None,
                        )
                        pipeline_states.append(state)

            if not use_mujoco:
                print("  Falling back to Brax pipeline reconstruction (this can take a bit)" )
                pipeline_fn = jax.jit(env.pipeline_init)
                t0 = time.time()
                for i in range(len(downsampled)):
                    state = pipeline_fn(jp.array(q_np[i]), jp.array(qd_np[i]))
                    pipeline_states.append(state)
                    if i % 20 == 0:
                        print(f"    Progress: {i}/{len(downsampled)}", end='\r')
                print(f"\n    Reconstruction took {time.time() - t0:.2f}s")

            # Render to HTML
            html_path = os.path.join(
                os.path.dirname(args.data_path),
                f'episode_{args.episode_idx}_visualization.html'
            )
            print("  Rendering HTML...")
            t0 = time.time()
            html_string = html.render(
                env.sys.tree_replace({'opt.timestep': env.dt * downsample_factor}),
                pipeline_states,
                height=1080,
                colab=False
            )
            print(f"    Rendering took {time.time() - t0:.2f}s")

            with open(html_path, 'w') as f:
                f.write(html_string)

            print(f"✓ HTML visualization saved to {html_path}")
            print("  Open it in a web browser to view the interactive 3D animation")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()




"""
Usage Examples:

# Basic inspection:
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz

# With plots:
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --plot

# With interactive HTML visualization (default 5x downsample):
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --html

# Higher fidelity HTML (2x downsample):
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --html --downsample 2

# Everything together:
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --plot --html

# Inspect different episode:
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --html --episode_idx 5

# Fast preview (10x downsample):
python inspect_trajectories.py --data_path logs/ppo_policy/trajectories/trajectories.npz --html --downsample 10
"""
