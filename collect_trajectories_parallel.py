#!/usr/bin/env python3
"""
PARALLEL trajectory collection using JAX vmap for massive speedup.

Collects multiple episodes simultaneously on GPU.
"""

import argparse
import os
import time

import jax
from jax import numpy as jp
import numpy as np
import mediapy as media

from brax import envs
from brax.io import model

from go2_env_brax import Go2Env


def main():
    parser = argparse.ArgumentParser(description="Collect expert trajectories in parallel")
    parser.add_argument("-e", "--exp_name", type=str, required=True)
    parser.add_argument("--n_episodes", type=int, default=10000)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Number of parallel episodes (default: 1024)")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    log_dir = f"logs/{args.exp_name}"
    model_path = f"{log_dir}/policy"
    
    if args.output_dir is None:
        args.output_dir = f"{log_dir}/trajectories"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ENV_DT = 0.02
    if args.duration is not None:
        args.max_steps = int(args.duration / ENV_DT)
    
    print("=" * 80)
    print("PARALLEL Go2 Trajectory Collection")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Steps per episode: {args.max_steps} ({args.max_steps * ENV_DT:.1f} sec)")
    print(f"Batch size (parallel): {args.batch_size}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # Load model
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        return
    
    print("\nLoading policy...")
    params = model.load_params(model_path)
    
    # Setup environment
    envs.register_environment('go2', Go2Env)
    env = envs.get_environment('go2')
    
    # Create inference function
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
    
    print("✓ Policy loaded")
    print(f"✓ Environment created")
    
    # Create vectorized rollout function
    def rollout_episode(rng):
        """Run a single episode and return trajectory data."""
        # Reset
        reset_rng, cmd_rng, rollout_rng = jax.random.split(rng, 3)
        state = env.reset(reset_rng)
        
        # Sample command
        command = env.sample_command(cmd_rng)
        state.info['command'] = command
        
        # Storage for trajectory
        def step_fn(carry, _):
            state, rng = carry
            act_rng, rng = jax.random.split(rng)
            
            # Get action
            action, _ = inference_fn(state.obs, act_rng)
            
            # Extract state and conditioning
            full_state = env.get_full_state(state.pipeline_state)
            conditioning = env.get_conditioning_vector(state.pipeline_state, state.info['command'])
            
            # Step
            next_state = env.step(state, action)
            
            return (next_state, rng), (full_state, conditioning, action, next_state.done)
        
        # Run trajectory
        _, (states, conditionings, actions, dones) = jax.lax.scan(
            step_fn, (state, rollout_rng), None, length=args.max_steps
        )
        
        # Check if episode completed without early termination
        episode_done = jp.any(dones)
        
        return states, conditionings, actions, episode_done
    
    # Vectorize over batch
    vmap_rollout = jax.jit(jax.vmap(rollout_episode))
    
    print("\n🔥 Compiling parallel rollout (this may take a minute)...")
    start_compile = time.time()
    
    # Warmup compilation
    test_rngs = jax.random.split(jax.random.PRNGKey(0), args.batch_size)
    _ = vmap_rollout(test_rngs)
    
    compile_time = time.time() - start_compile
    print(f"✓ Compiled in {compile_time:.1f}s")
    
    # Collect trajectories in batches
    all_states = []
    all_conditionings = []
    all_actions = []
    
    rng = jax.random.PRNGKey(args.seed)
    collected = 0
    total_generated = 0
    
    print(f"\nCollecting {args.n_episodes} successful episodes...")
    print("(Only keeping episodes that complete without early termination)\n")
    
    start_time = time.time()
    
    while collected < args.n_episodes:
        # Generate batch
        batch_rngs = jax.random.split(rng, args.batch_size + 1)
        rng = batch_rngs[0]
        episode_rngs = batch_rngs[1:]
        
        # Run parallel rollouts
        batch_states, batch_cond, batch_actions, batch_done = vmap_rollout(episode_rngs)
        
        # Filter out episodes that terminated early
        success_mask = ~batch_done  # True for episodes that completed
        n_success = jp.sum(success_mask).item()
        total_generated += args.batch_size
        
        if n_success > 0:
            # Extract successful episodes
            success_indices = jp.where(success_mask)[0]
            
            for idx in success_indices:
                if collected >= args.n_episodes:
                    break
                
                all_states.append(np.array(batch_states[idx]))
                all_conditionings.append(np.array(batch_cond[idx]))
                all_actions.append(np.array(batch_actions[idx]))
                collected += 1
        
        # Progress update
        elapsed = time.time() - start_time
        success_rate = collected / total_generated * 100
        eps_per_sec = total_generated / elapsed if elapsed > 0 else 0
        
        print(f"  Collected: {collected}/{args.n_episodes} | "
              f"Generated: {total_generated} | "
              f"Success: {success_rate:.1f}% | "
              f"Speed: {eps_per_sec:.1f} eps/sec | "
              f"Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    
    # Save
    print(f"\n{'=' * 80}")
    print("Collection Complete!")
    print(f"{'=' * 80}")
    print(f"✓ Collected {collected} successful episodes")
    print(f"  Total generated: {total_generated}")
    print(f"  Success rate: {collected/total_generated*100:.1f}%")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Speed: {total_generated/total_time:.1f} episodes/sec")
    
    # Verify lengths
    lengths = [len(s) for s in all_states]
    assert all(l == args.max_steps for l in lengths), "Not all episodes have correct length!"
    print(f"  ✓ All episodes have exactly {args.max_steps} steps")
    
    # Save
    output_path = f"{args.output_dir}/trajectories.npz"
    np.savez_compressed(
        output_path,
        states=np.array(all_states, dtype=object),
        conditionings=np.array(all_conditionings, dtype=object),
        actions=np.array(all_actions, dtype=object),
        episode_lengths=np.array(lengths),
    )
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    print(f"\nData shapes:")
    print(f"  States: {all_states[0].shape} (37D)")
    print(f"  Conditioning: {all_conditionings[0].shape} (36D)")
    print(f"  Actions: {all_actions[0].shape} (12D)")


if __name__ == "__main__":
    main()


"""
Usage:

# Collect 10000 episodes in parallel (FAST!):
python collect_trajectories_parallel.py --exp_name ppo_policy

# Smaller batch for limited GPU memory:
python collect_trajectories_parallel.py --exp_name ppo_policy --batch_size 512

# Collect fewer for testing:
python collect_trajectories_parallel.py --exp_name ppo_policy --n_episodes 1000

Expected speedup:
- Serial script: ~1-10 episodes/sec
- Parallel script: ~100-1000+ episodes/sec (100x faster!)
"""
