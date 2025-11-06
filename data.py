"""
Data loading utilities for state-only trajectory datasets.

Supports loading from CSV, NPZ, and NPY files with sliding window extraction.
"""

import os
import glob
import numpy as np
import pandas as pd
import jax.numpy as jnp
from typing import List, Optional, Tuple, Iterator
from dataclasses import dataclass


@dataclass
class TrajectoryDataset:
    """
    Dataset of state-only trajectory windows for DynaFlow training.
    
    Loads trajectories from files and extracts sliding windows of length H+1.
    Optionally includes conditioning vectors aligned with each window.
    """
    
    data: np.ndarray  # (N, H+1, state_dim)
    cond_data: Optional[np.ndarray] = None  # (N, cond_dim) if available
    
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        """Get a single trajectory window, optionally with conditioning."""
        if self.cond_data is not None:
            return self.data[idx], self.cond_data[idx]
        return (self.data[idx],)
    
    def get_batch(self, indices: np.ndarray) -> dict:
        """Get a batch of trajectories as dictionary."""
        batch = {'trajectories': self.data[indices]}
        if self.cond_data is not None:
            batch['cond'] = self.cond_data[indices]
        return batch
    
    @property
    def state_dim(self) -> int:
        return self.data.shape[-1]
    
    @property
    def horizon(self) -> int:
        return self.data.shape[1] - 1
    
    @property
    def cond_dim(self) -> Optional[int]:
        return self.cond_data.shape[-1] if self.cond_data is not None else None


def load_trajectory_dataset(
    paths: List[str],
    horizon: int = 16,
    stride: int = 1,
    state_columns: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> TrajectoryDataset:
    """
    Load state-only trajectory dataset from files.
    
    Args:
        paths: List of file/directory paths to load from
        horizon: Trajectory horizon H (windows will be H+1 long)
        stride: Stride for sliding window extraction
        state_columns: Column names for CSV files
        max_files: Maximum number of files to load
        max_windows: Maximum number of windows to extract (for memory management)
    
    Returns:
        TrajectoryDataset instance
    """
    windows = []
    cond_windows = []
    has_cond = None
    cond_dim = None
    
    # Collect all files
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.csv"), recursive=True)))
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.npz"), recursive=True)))
            files.extend(sorted(glob.glob(os.path.join(p, "**", "*.npy"), recursive=True)))
        else:
            files.append(p)
    
    if max_files is not None:
        files = files[:max_files]
    
    # Load each file
    for filepath in files:
        if max_windows is not None and len(windows) >= max_windows:
            print(f"Reached max_windows limit ({max_windows}), stopping data loading")
            break
            
        if filepath.endswith(".csv"):
            arr, cond_arr = _load_csv(filepath, state_columns)
        elif filepath.endswith(".npz"):
            arr, cond_arr = _load_npz(filepath)
        elif filepath.endswith(".npy"):
            arr, cond_arr = _load_npy(filepath)
        else:
            continue
        
        # Process trajectories (arr might be ragged or batched)
        if isinstance(arr, list):
            # Ragged array of episodes
            for i, traj in enumerate(arr):
                if max_windows is not None and len(windows) >= max_windows:
                    break
                cond_ep = cond_arr[i] if cond_arr is not None else None
                _extract_windows(traj, cond_ep, horizon, stride, windows, cond_windows)
                if has_cond is None and cond_ep is not None:
                    has_cond = True
                    cond_dim = cond_ep.shape[-1]
        elif arr.ndim == 2:
            # Single trajectory (T, state_dim)
            _extract_windows(arr, cond_arr, horizon, stride, windows, cond_windows)
            if has_cond is None and cond_arr is not None:
                has_cond = True
                cond_dim = cond_arr.shape[-1]
        elif arr.ndim == 3:
            # Batched trajectories (N, T, state_dim)
            for i in range(arr.shape[0]):
                if max_windows is not None and len(windows) >= max_windows:
                    break
                cond_ep = cond_arr[i] if cond_arr is not None else None
                _extract_windows(arr[i], cond_ep, horizon, stride, windows, cond_windows)
                if has_cond is None and cond_ep is not None:
                    has_cond = True
                    cond_dim = cond_ep.shape[-1]
    
    if len(windows) == 0:
        raise RuntimeError("No trajectory windows found. Check paths and horizon.")
    
    # Limit total windows if specified
    if max_windows is not None and len(windows) > max_windows:
        windows = windows[:max_windows]
        if cond_windows:
            cond_windows = cond_windows[:max_windows]
    
    # Stack windows - ensure proper dtype
    # Convert all windows to numpy arrays first to avoid object dtype
    windows = [np.asarray(w, dtype=np.float32) for w in windows]
    data = np.stack(windows, axis=0)  # (N, H+1, state_dim)
    
    cond_data = None
    if has_cond and len(cond_windows) > 0:
        cond_windows = [np.asarray(c, dtype=np.float32) for c in cond_windows]
        cond_data = np.stack(cond_windows, axis=0)  # (N, cond_dim)
    
    return TrajectoryDataset(data=data, cond_data=cond_data)


def _load_csv(filepath: str, state_columns: Optional[List[str]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load trajectory from CSV file."""
    df = pd.read_csv(filepath)
    if state_columns is None:
        raise ValueError("state_columns must be provided for CSV files")
    arr = df[state_columns].to_numpy(dtype=np.float32)
    return arr, None


def _load_npy(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load trajectory from NPY file."""
    arr = np.load(filepath, allow_pickle=True)
    if arr.dtype == object:
        # Ragged array
        return list(arr), None
    return arr.astype(np.float32), None


def _load_npz(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load trajectories from NPZ file (supports DDAT-style format)."""
    data = np.load(filepath, allow_pickle=True)

    # Check for parallel collection format (states + conditionings)
    if "states" in data and "conditionings" in data:
        traj_data = data["states"]
        cond_data = data["conditionings"]
        
        # Handle object arrays (ragged)
        if isinstance(traj_data, np.ndarray) and traj_data.dtype == object:
            traj_list = [np.asarray(t, dtype=np.float32) for t in traj_data]
            if cond_data is not None and isinstance(cond_data, np.ndarray) and cond_data.dtype == object:
                cond_list = [np.asarray(c, dtype=np.float32) for c in cond_data]
            elif cond_data is not None:
                cond_list = [np.asarray(cond_data, dtype=np.float32)]
            else:
                cond_list = None
            return traj_list, cond_list
        else:
            traj_converted = np.asarray(traj_data, dtype=np.float32)
            cond_converted = np.asarray(cond_data, dtype=np.float32) if cond_data is not None else None
            return traj_converted, cond_converted
    
    # # Check for states-only format
    # if "states" in data:
    #     traj_data = data["states"]
    #     if isinstance(traj_data, np.ndarray) and traj_data.dtype == object:
    #         traj_list = [np.asarray(t, dtype=np.float32) for t in traj_data]
    #         return traj_list, None
    #     else:
    #         return np.asarray(traj_data, dtype=np.float32), None
    
    # Fallback: use first key
    key = list(data.keys())[0]
    arr = data[key]
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr_list = [np.asarray(item, dtype=np.float32) for item in arr]
        return arr_list, None
    return np.asarray(arr, dtype=np.float32), None


def _extract_windows(
    traj: np.ndarray,
    cond_traj: Optional[np.ndarray],
    horizon: int,
    stride: int,
    windows: List,
    cond_windows: List
):
    """Extract sliding windows from a single trajectory."""
    # Ensure trajectory is a numpy array with proper dtype
    if not isinstance(traj, np.ndarray):
        traj = np.asarray(traj, dtype=np.float32)
    elif traj.dtype == object or traj.dtype != np.float32:
        traj = np.asarray(traj, dtype=np.float32)
    
    T = traj.shape[0]
    H1 = horizon + 1
    
    if T < H1:
        return
    
    for s in range(0, T - H1 + 1, stride):
        window = traj[s:s + H1]
        # Ensure window is properly typed
        window = np.asarray(window, dtype=np.float32)
        windows.append(window)
        
        if cond_traj is not None:
            # Use conditioning at window start time
            cond_window = cond_traj[s]
            if not isinstance(cond_window, np.ndarray):
                cond_window = np.asarray(cond_window, dtype=np.float32)
            elif cond_window.dtype == object or cond_window.dtype != np.float32:
                cond_window = np.asarray(cond_window, dtype=np.float32)
            cond_windows.append(cond_window)


def create_data_iterator(
    dataset: TrajectoryDataset,
    batch_size: int,
    shuffle: bool = True,
    rng: Optional[np.random.Generator] = None
) -> Iterator[dict]:
    """
    Create an iterator over batches from the dataset.
    
    Args:
        dataset: TrajectoryDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        rng: Random number generator
    
    Yields:
        Batches as dictionaries with 'trajectories' and optionally 'cond'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = len(dataset)
    indices = np.arange(N)
    
    if shuffle:
        rng.shuffle(indices)
    
    for i in range(0, N, batch_size):
        batch_indices = indices[i:i + batch_size]
        if len(batch_indices) < batch_size:
            continue  # Drop last incomplete batch
        yield dataset.get_batch(batch_indices)


def prepare_batch_for_training(
    batch: dict,
    rng: np.random.Generator,
    noise_scale: float = 1.0
) -> dict:
    """
    Prepare a batch for training by sampling t and generating x0.
    
    Args:
        batch: Dictionary with 'trajectories' and optionally 'cond'
        rng: Random number generator
        noise_scale: Scale of Gaussian noise for x0
    
    Returns:
        Dictionary with x0, x1, t, and optionally cond
    """
    x1 = batch['trajectories']  # (batch, H+1, state_dim)
    B, H1, D = x1.shape
    
    # Sample time t ~ U(0, 1)
    t = rng.uniform(0.0, 1.0, size=(B, 1)).astype(np.float32)
    
    # Sample x0 ~ N(0, σ²) with per-sample variance
    std = x1.std(axis=(1, 2), keepdims=True)
    std = np.maximum(std, 1e-3)
    x0 = rng.normal(0.0, 1.0, size=x1.shape).astype(np.float32) * std * noise_scale
    
    result = {
        'x0': x0,
        'x1': x1,
        't': t,
    }
    
    if 'cond' in batch:
        result['cond'] = batch['cond']
    
    return result


def numpy_to_jax(batch: dict) -> dict:
    """Convert numpy arrays in batch to JAX arrays."""
    return {k: jnp.array(v) for k, v in batch.items()}
