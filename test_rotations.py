"""
Test script to verify rotation functions match between Genesis and MuJoCo wrapper
"""

import torch
import numpy as np


def genesis_inv_quat(quat):
    """From genesis/utils/geom.py"""
    if isinstance(quat, torch.Tensor):
        scaling = torch.tensor([1, -1, -1, -1], device=quat.device)
        return quat * scaling
    elif isinstance(quat, np.ndarray):
        scaling = np.array([1, -1, -1, -1], dtype=quat.dtype)
        return quat * scaling


def genesis_transform_by_quat(v, quat):
    """From genesis/utils/geom.py - torch version"""
    if isinstance(v, torch.Tensor) and isinstance(quat, torch.Tensor):
        qvec = quat[..., 1:]
        t = torch.cross(qvec, v, dim=-1) * 2
        return v + quat[..., :1] * t + torch.cross(qvec, t, dim=-1)


def mujoco_wrapper_inv_quat(quat):
    """From our env_wrapper.py"""
    scaling = torch.tensor([1, -1, -1, -1], device=quat.device, dtype=quat.dtype)
    return quat * scaling


def mujoco_wrapper_transform_by_quat(v, quat):
    """From our env_wrapper.py"""
    qvec = quat[..., 1:]  # [x, y, z]
    t = torch.cross(qvec, v, dim=-1) * 2
    return v + quat[..., :1] * t + torch.cross(qvec, t, dim=-1)


def test_rotations():
    print("="*60)
    print("Testing Quaternion Rotation Functions")
    print("="*60)
    
    # Test 1: Identity quaternion
    print("\n[Test 1] Identity quaternion")
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    vec = torch.tensor([[1.0, 0.0, 0.0]])
    
    gen_result = genesis_transform_by_quat(vec, quat)
    muj_result = mujoco_wrapper_transform_by_quat(vec, quat)
    
    print(f"Input vector: {vec}")
    print(f"Quaternion: {quat}")
    print(f"Genesis result: {gen_result}")
    print(f"MuJoCo wrapper result: {muj_result}")
    print(f"Match: {torch.allclose(gen_result, muj_result)}")
    
    # Test 2: 90 degree rotation around z-axis
    print("\n[Test 2] 90° rotation around z-axis")
    angle = np.pi / 2
    quat = torch.tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]], dtype=torch.float32)
    vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    
    gen_result = genesis_transform_by_quat(vec, quat)
    muj_result = mujoco_wrapper_transform_by_quat(vec, quat)
    
    print(f"Input vector: {vec}")
    print(f"Quaternion (90° z): {quat}")
    print(f"Genesis result: {gen_result}")
    print(f"MuJoCo wrapper result: {muj_result}")
    print(f"Expected: [[0, 1, 0]] (approx)")
    print(f"Match: {torch.allclose(gen_result, muj_result)}")
    
    # Test 3: Inverse quaternion
    print("\n[Test 3] Inverse quaternion")
    quat = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32)  # 90° around z
    
    gen_inv = genesis_inv_quat(quat)
    muj_inv = mujoco_wrapper_inv_quat(quat)
    
    print(f"Original quat: {quat}")
    print(f"Genesis inverse: {gen_inv}")
    print(f"MuJoCo wrapper inverse: {muj_inv}")
    print(f"Match: {torch.allclose(gen_inv, muj_inv)}")
    
    # Test 4: Inverse rotation (should undo rotation)
    print("\n[Test 4] Inverse rotation (undo 90° rotation)")
    quat = torch.tensor([[0.7071, 0.0, 0.0, 0.7071]], dtype=torch.float32)
    vec = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    
    # Rotate
    rotated_gen = genesis_transform_by_quat(vec, quat)
    rotated_muj = mujoco_wrapper_transform_by_quat(vec, quat)
    
    # Unrotate with inverse
    inv_quat_gen = genesis_inv_quat(quat)
    inv_quat_muj = mujoco_wrapper_inv_quat(quat)
    
    restored_gen = genesis_transform_by_quat(rotated_gen, inv_quat_gen)
    restored_muj = mujoco_wrapper_transform_by_quat(rotated_muj, inv_quat_muj)
    
    print(f"Original vector: {vec}")
    print(f"After rotation: {rotated_gen}")
    print(f"After inverse rotation (Genesis): {restored_gen}")
    print(f"After inverse rotation (MuJoCo): {restored_muj}")
    print(f"Restored match original: {torch.allclose(restored_gen, vec, atol=1e-5)} / {torch.allclose(restored_muj, vec, atol=1e-5)}")
    
    # Test 5: Batch operations
    print("\n[Test 5] Batch operations")
    quats = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # identity
        [0.7071, 0.0, 0.0, 0.7071],  # 90° z
        [0.7071, 0.0, 0.7071, 0.0],  # 90° y
    ])
    vecs = torch.tensor([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    
    gen_results = genesis_transform_by_quat(vecs, quats)
    muj_results = mujoco_wrapper_transform_by_quat(vecs, quats)
    
    print(f"Genesis results:\n{gen_results}")
    print(f"MuJoCo wrapper results:\n{muj_results}")
    print(f"All match: {torch.allclose(gen_results, muj_results, atol=1e-5)}")
    
    print("\n" + "="*60)
    print("All rotation tests PASSED ✓" if torch.allclose(gen_results, muj_results, atol=1e-5) else "FAILED ✗")
    print("="*60)


if __name__ == "__main__":
    test_rotations()
