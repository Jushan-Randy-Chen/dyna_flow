"""
JAX/Flax implementation of the 1D Diffusion Transformer (DiT) for action prediction.

This module implements the action prediction network D_θ that takes a noisy state
trajectory X_t, time t, and optional conditioning c, and outputs an action sequence U^.

Based on the DiT architecture from the DynaFlow paper.
"""

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from typing import Optional, Callable
import math


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift"""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))"""
    return x * jnp.tanh(nn.softplus(x))


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for sequence positions."""
    
    dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Position indices (seq_len,)
        Returns:
            Positional embeddings (seq_len, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class TimeEmbedding(nn.Module):
    """MLP-based time embedding for diffusion timestep t."""
    
    dim: int
    
    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            t: Time values (batch, 1)
        Returns:
            Time embeddings (batch, dim)
        """
        x = nn.Dense(self.dim)(t)
        x = mish(x)  # Exact match to PyTorch nn.Mish()
        x = nn.Dense(self.dim)(x)
        return x


class ContinuousCondEmbedder(nn.Module):
    """Embed continuous conditioning attributes using attention.
    
    Modified from PyTorch reference to embed continuous variables.
    Matches the implementation from DiT/DiT.py.
    """
    
    attr_dim: int
    hidden_size: int
    
    @nn.compact
    def __call__(self, attr: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            attr: (batch, attr_dim) continuous attributes
            mask: (batch, attr_dim) binary mask (0 = ignore)
        Returns:
            Embedding (batch, hidden_size)
        """
        # Project each attribute dimension: (batch, attr_dim) -> (batch, attr_dim * 128)
        # Then reshape to (batch, attr_dim, 128)
        emb = nn.Dense(self.attr_dim * 128)(attr)  # (batch, attr_dim * 128)
        emb = emb.reshape((-1, self.attr_dim, 128))  # (batch, attr_dim, 128)
        
        if mask is not None:
            emb = emb * mask[:, :, None]
        
        # Self-attention over attribute dimensions
        emb = nn.MultiHeadDotProductAttention(
            num_heads=2,
            qkv_features=128,
            deterministic=True
        )(emb, emb)  # (batch, attr_dim, 128)
        
        # Flatten and project to hidden size (matches PyTorch version)
        emb = emb.reshape((-1, self.attr_dim * 128))  # (batch, attr_dim * 128)
        return nn.Dense(self.hidden_size)(emb)  # (batch, hidden_size)


class DiTBlock(nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""
    
    hidden_size: int
    n_heads: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            x: Input tokens (batch, seq_len, hidden_size)
            t: Time embedding (batch, hidden_size)
            deterministic: Whether to apply dropout
        Returns:
            Output tokens (batch, seq_len, hidden_size)
        """
        # adaLN modulation parameters
        modulation = nn.Dense(self.hidden_size * 6)(nn.silu(t))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )
        
        # Multi-head self-attention with adaLN
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        mod_x = modulate(norm_x, shift_msa, scale_msa)
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout if not deterministic else 0.0,
            deterministic=deterministic
        )(mod_x, mod_x)
        x = x + gate_msa[:, None, :] * attn_out
        
        # MLP with adaLN
        norm_x2 = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        mod_x2 = modulate(norm_x2, shift_mlp, scale_mlp)
        mlp_out = nn.Dense(self.hidden_size * 4)(mod_x2)
        mlp_out = nn.gelu(mlp_out, approximate=True)  # GELU with tanh approximation
        mlp_out = nn.Dropout(rate=self.dropout, deterministic=deterministic)(mlp_out)
        mlp_out = nn.Dense(self.hidden_size)(mlp_out)
        x = x + gate_mlp[:, None, :] * mlp_out
        
        return x


class FinalLayer1d(nn.Module):
    """Final layer with adaptive layer norm modulation."""
    
    hidden_size: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tokens (batch, seq_len, hidden_size)
            t: Time embedding (batch, hidden_size)
        Returns:
            Output (batch, seq_len, out_dim)
        """
        modulation = nn.Dense(self.hidden_size * 2)(nn.silu(t))
        shift, scale = jnp.split(modulation, 2, axis=-1)
        
        norm_x = nn.LayerNorm(use_bias=False, use_scale=False, epsilon=1e-6)(x)
        mod_x = modulate(norm_x, shift, scale)
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros)(mod_x)


class DiT1d(nn.Module):
    """1D Diffusion Transformer for sequence modeling.
    
    This model processes sequences with self-attention and is conditioned on
    time and optional attributes.
    """
    
    x_dim: int
    d_model: int = 384
    n_heads: int = 6
    depth: int = 6
    dropout: float = 0.1
    attr_dim: Optional[int] = None
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        attr: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            x: Input sequence (batch, seq_len, x_dim)
            t: Time (batch, 1)
            attr: Optional attributes (batch, attr_dim)
            mask: Optional attribute mask (batch, attr_dim)
            deterministic: Whether to apply dropout
        Returns:
            Output sequence (batch, seq_len, x_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = nn.Dense(self.d_model)(x)
        
        # Add positional embeddings
        pos_emb = SinusoidalPosEmb(self.d_model)(jnp.arange(seq_len))
        x = x + pos_emb[None, :, :]
        
        # Time embedding
        t_emb = TimeEmbedding(self.d_model)(t)
        
        # Add attribute conditioning if provided
        if attr is not None:
            assert self.attr_dim is not None, "Model is not conditional"
            attr_emb = ContinuousCondEmbedder(self.attr_dim, self.d_model)(attr, mask)
            t_emb = t_emb + attr_emb
        
        # Apply transformer blocks
        for _ in range(self.depth):
            x = DiTBlock(self.d_model, self.n_heads, self.dropout)(x, t_emb, deterministic)
        
        # Final layer
        x = FinalLayer1d(self.d_model, self.x_dim)(x, t_emb)
        
        return x


class ActionPredictor(nn.Module):
    """
    Action prediction network D_θ for DynaFlow.
    
    Takes a noisy state trajectory X_t, time t, and optional conditioning c,
    and outputs an action sequence U^.
    """
    
    state_dim: int
    action_dim: int
    d_model: int = 384
    n_heads: int = 6
    depth: int = 3
    dropout: float = 0.1
    cond_dim: Optional[int] = None
    
    @nn.compact
    def __call__(
        self,
        X_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            X_t: Noisy state trajectory (batch, H+1, state_dim)
            t: Time (batch, 1)
            cond: Optional conditioning (batch, cond_dim)
            deterministic: Whether to apply dropout
        Returns:
            U_hat: Predicted actions (batch, H, action_dim)
        """
        # Run DiT backbone on the full state trajectory
        y = DiT1d(
            x_dim=self.state_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            depth=self.depth,
            dropout=self.dropout,
            attr_dim=self.cond_dim
        )(X_t, t, attr=cond, deterministic=deterministic)
        
        # Project each state token to action dimension
        a_tokens = nn.Dense(self.action_dim)(y)
        
        # Use actions for steps 1..H (drop the first token which corresponds to x0)
        U_hat = a_tokens[:, 1:, :]
        
        return U_hat


def create_action_predictor(
    state_dim: int,
    action_dim: int,
    d_model: int = 384,
    n_heads: int = 6,
    depth: int = 6,
    cond_dim: Optional[int] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> tuple:
    """
    Create and initialize an ActionPredictor model.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        d_model: Hidden dimension
        n_heads: Number of attention heads
        depth: Number of transformer blocks
        cond_dim: Optional conditioning dimension
        rng: Random key for initialization
    
    Returns:
        (model, params) tuple
    """
    if rng is None:
        rng = random.PRNGKey(0)
    
    model = ActionPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        cond_dim=cond_dim
    )
    
    # Initialize with dummy inputs
    dummy_X_t = jnp.ones((1, 17, state_dim))  # H+1 = 17 for horizon 16
    dummy_t = jnp.ones((1, 1))
    dummy_cond = jnp.ones((1, cond_dim)) if cond_dim else None
    
    # Split RNG for params and dropout
    init_rng, dropout_rng = random.split(rng)
    params = model.init({'params': init_rng, 'dropout': dropout_rng}, dummy_X_t, dummy_t, cond=dummy_cond, deterministic=True)
    
    return model, params


def count_parameters(params) -> int:
    """Count the number of parameters in a model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
