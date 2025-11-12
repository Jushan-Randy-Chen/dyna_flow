"""
JAX/Flax NNX implementation of the 1D Diffusion Transformer (DiT) for action prediction.

This module builds the action prediction network D_θ that consumes a noisy state
trajectory X_t, diffusion time t, and optional conditioning signal c, and returns
the predicted action sequence Ũ. The architecture mirrors the DiT backbone from
the DynaFlow paper but is written using the object-style Flax.nnx API that is also
used throughout the `gpc` reference repository.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx, serialization
from flax.nnx import statelib


def modulate(x: jnp.ndarray, shift: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    """Apply adaptive layer norm modulation: x * (1 + scale) + shift."""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation: x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))."""
    return x * jnp.tanh(nnx.softplus(x))


class SinusoidalPosEmb(nnx.Module):
    """Sinusoidal positional embeddings for sequence positions."""

    def __init__(self, dim: int):
        self.dim = dim
        self.half_dim = max(dim // 2, 1)

    def __call__(self, positions: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            positions: Position indices (seq_len,)
        Returns:
            Positional embeddings (seq_len, dim)
        """
        freq_factor = math.log(10000.0) / max(self.half_dim - 1, 1)
        freqs = jnp.exp(jnp.arange(self.half_dim) * -freq_factor)
        # Ensure float input for broadcasting.
        angles = positions.astype(jnp.float32)[..., None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        if emb.shape[-1] < self.dim:
            pad_width = self.dim - emb.shape[-1]
            emb = jnp.pad(emb, ((0, 0), (0, pad_width)))
        return emb


class TimeEmbedding(nnx.Module):
    """MLP-based time embedding for diffusion timestep t."""

    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(1, dim, rngs=rngs)
        self.fc2 = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            t: Time values (batch, 1)
        Returns:
            Time embeddings (batch, dim)
        """
        x = self.fc1(t)
        x = mish(x)
        x = self.fc2(x)
        return x


class ContinuousCondEmbedder(nnx.Module):
    """Embed continuous conditioning attributes using attention."""

    def __init__(self, attr_dim: int, hidden_size: int, rngs: nnx.Rngs):
        self.attr_dim = attr_dim
        self.hidden_size = hidden_size
        self.embed = nnx.Linear(attr_dim, attr_dim * 128, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=2,
            in_features=128,
            qkv_features=128,
            out_features=128,
            dropout_rate=0.0,
            rngs=rngs,
        )
        self.out_proj = nnx.Linear(attr_dim * 128, hidden_size, rngs=rngs)

    def __call__(
        self, attr: jnp.ndarray, mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Args:
            attr: (batch, attr_dim) or (batch, seq_len, attr_dim) attributes.
            mask: Optional mask matching attr's leading dimensions.
        Returns:
            Embedding of shape (batch, hidden_size) for 2D input or
            (batch, seq_len, hidden_size) for time-varying conditioning.
        """
        if attr.shape[-1] != self.attr_dim:
            raise ValueError(f"Expected attr_dim={self.attr_dim}, got {attr.shape[-1]}")

        has_time_axis = attr.ndim == 3
        if has_time_axis:
            batch, seq_len, _ = attr.shape
            attr_flat = attr.reshape((-1, self.attr_dim))
            mask_flat = None
            if mask is not None:
                mask_flat = mask.reshape((-1, self.attr_dim))
        else:
            batch = attr.shape[0]
            seq_len = None
            attr_flat = attr
            mask_flat = mask

        emb = self.embed(attr_flat)
        emb = emb.reshape((-1, self.attr_dim, 128))

        if mask_flat is not None:
            emb = emb * mask_flat[:, :, None]

        emb = self.attn(emb, deterministic=True, decode=False)
        emb = emb.reshape((-1, self.attr_dim * 128))
        emb = self.out_proj(emb)

        if has_time_axis:
            emb = emb.reshape((batch, seq_len, self.hidden_size))

        return emb


class DiTBlock(nnx.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.modulation = nnx.Linear(hidden_size, hidden_size * 6, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=n_heads,
            in_features=hidden_size,
            qkv_features=hidden_size,
            out_features=hidden_size,
            dropout_rate=dropout,
            rngs=rngs,
        )
        self.norm_msa = nnx.LayerNorm(
            num_features=hidden_size,
            use_bias=False,
            use_scale=False,
            epsilon=1e-6,
            rngs=rngs,
        )
        self.norm_mlp = nnx.LayerNorm(
            num_features=hidden_size,
            use_bias=False,
            use_scale=False,
            epsilon=1e-6,
            rngs=rngs,
        )
        self.mlp_fc1 = nnx.Linear(hidden_size, hidden_size * 4, rngs=rngs)
        self.dropout_layer = nnx.Dropout(rate=dropout, rngs=rngs)
        self.mlp_fc2 = nnx.Linear(hidden_size * 4, hidden_size, rngs=rngs)

    def __call__(
        self, x: jnp.ndarray, t: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """
        Args:
            x: Input tokens (batch, seq_len, hidden_size)
            t: Time embedding (batch, hidden_size)
            deterministic: Whether to apply dropout
        Returns:
            Output tokens (batch, seq_len, hidden_size)
        """
        modulation = self.modulation(nnx.silu(t))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )

        norm_x = self.norm_msa(x)
        mod_x = modulate(norm_x, shift_msa, scale_msa)
        attn_out = self.attn(mod_x, deterministic=deterministic, decode=False)
        x = x + gate_msa[:, None, :] * attn_out

        norm_x2 = self.norm_mlp(x)
        mod_x2 = modulate(norm_x2, shift_mlp, scale_mlp)
        mlp_out = self.mlp_fc1(mod_x2)
        mlp_out = nnx.gelu(mlp_out, approximate=True)
        mlp_out = self.dropout_layer(mlp_out, deterministic=deterministic)
        mlp_out = self.mlp_fc2(mlp_out)
        x = x + gate_mlp[:, None, :] * mlp_out

        return x


class FinalLayer1d(nnx.Module):
    """Final layer with adaptive layer norm modulation."""

    def __init__(self, hidden_size: int, out_dim: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.modulation = nnx.Linear(hidden_size, hidden_size * 2, rngs=rngs)
        self.norm = nnx.LayerNorm(
            num_features=hidden_size,
            use_bias=False,
            use_scale=False,
            epsilon=1e-6,
            rngs=rngs,
        )
        self.out = nnx.Linear(
            hidden_size,
            out_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: Input tokens (batch, seq_len, hidden_size)
            t: Time embedding (batch, hidden_size)
        Returns:
            Output (batch, seq_len, out_dim)
        """
        modulation = self.modulation(nnx.silu(t))
        shift, scale = jnp.split(modulation, 2, axis=-1)
        norm_x = self.norm(x)
        mod_x = modulate(norm_x, shift, scale)
        return self.out(mod_x)


class DiT1d(nnx.Module):
    """1D Diffusion Transformer for sequence modeling."""

    def __init__(
        self,
        x_dim: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float,
        attr_dim: Optional[int],
        rngs: nnx.Rngs,
    ):
        self.x_dim = x_dim
        self.d_model = d_model
        self.depth = depth
        self.attr_dim = attr_dim

        self.input_proj = nnx.Linear(x_dim, d_model, rngs=rngs)
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.time_emb = TimeEmbedding(d_model, rngs=rngs)

        if attr_dim is not None:
            self.attr_embed = ContinuousCondEmbedder(attr_dim, d_model, rngs=rngs)
        else:
            self.attr_embed = None

        for i in range(depth):
            setattr(
                self,
                f"block_{i}",
                DiTBlock(d_model, n_heads, dropout, rngs=rngs),
            )

        self.final_layer = FinalLayer1d(d_model, x_dim, rngs=rngs)

    def __call__(
        self,
        x: jnp.ndarray,
        t: jnp.ndarray,
        attr: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            x: Input sequence (batch, seq_len, x_dim)
            t: Time (batch, 1)
            attr: Optional attributes (batch, attr_dim or batch, seq, attr_dim)
            mask: Optional attribute mask
            deterministic: Whether to apply dropout
        Returns:
            Output sequence (batch, seq_len, x_dim)
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)

        positions = jnp.arange(seq_len, dtype=jnp.float32)
        pos_emb = self.pos_emb(positions)
        x = x + pos_emb[None, :, :]

        t_emb = self.time_emb(t)

        if attr is not None:
            if self.attr_embed is None:
                raise ValueError("Model instantiated without attr_dim but attr provided.")
            attr_emb = self.attr_embed(attr, mask)
            if attr_emb.ndim == 2:
                t_emb = t_emb + attr_emb
            else:
                x = x + attr_emb
                t_emb = t_emb + jnp.mean(attr_emb, axis=1)

        for i in range(self.depth):
            block = getattr(self, f"block_{i}")
            x = block(x, t_emb, deterministic=deterministic)

        x = self.final_layer(x, t_emb)
        return x


class ActionPredictor(nnx.Module):
    """
    Action prediction network D_θ for DynaFlow.

    Takes a noisy state trajectory X_t, time t, and optional conditioning c,
    and outputs an action sequence Ũ.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int,
        n_heads: int,
        depth: int,
        dropout: float,
        cond_dim: Optional[int],
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.backbone = DiT1d(
            x_dim=state_dim,
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            dropout=dropout,
            attr_dim=cond_dim,
            rngs=rngs,
        )
        self.action_head = nnx.Linear(state_dim, action_dim, rngs=rngs)

    def __call__(
        self,
        X_t: jnp.ndarray,
        t: jnp.ndarray,
        cond: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Args:
            X_t: Noisy state trajectory (batch, H+1, state_dim)
            t: Time (batch, 1)
            cond: Optional conditioning (batch, cond_dim or batch, seq, cond_dim)
            mask: Optional conditioning mask
            deterministic: Whether to apply dropout
        Returns:
            U_hat: Predicted actions (batch, H, action_dim)
        """
        y = self.backbone(X_t, t, attr=cond, mask=mask, deterministic=deterministic)
        a_tokens = self.action_head(y)
        a_tokens = jnp.tanh(a_tokens)
        U_hat = a_tokens[:, 1:, :]
        return U_hat


@dataclass
class ActionPredictorHandle:
    """Callable handle that exposes a Linen-style `.apply` for nnx modules."""

    graphdef: nnx.GraphDef
    aux_state: nnx.State

    def apply(self, params: nnx.State, *args, **kwargs):
        module = nnx.merge(self.graphdef, params, self.aux_state)
        return module(*args, **kwargs)


def create_action_predictor(
    state_dim: int,
    action_dim: int,
    d_model: int = 384,
    n_heads: int = 6,
    depth: int = 3,
    horizon: int = 17,
    cond_dim: Optional[int] = None,
    dropout: float = 0.1,
    rng: Optional[jax.random.PRNGKey] = None,
) -> Tuple[ActionPredictorHandle, nnx.State]:
    """
    Create and initialize an ActionPredictor model.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        d_model: Hidden dimension
        n_heads: Number of attention heads
        depth: Number of transformer blocks
        horizon: Sequence length (H+1)
        cond_dim: Optional conditioning dimension
        dropout: Dropout probability inside DiT blocks
        rng: Random key for initialization

    Returns:
        (model_handle, params_state)
    """
    if rng is None:
        rng = random.PRNGKey(0)

    rngs = nnx.Rngs(rng)
    module = ActionPredictor(
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        dropout=dropout,
        cond_dim=cond_dim,
        rngs=rngs,
    )

    graphdef, params, aux_state = nnx.split(module, nnx.Param, nnx.RngState)
    model_handle = ActionPredictorHandle(graphdef=graphdef, aux_state=aux_state)

    # Force materialization via a dummy call to ensure arrays are initialized.
    dummy_X_t = jnp.ones((1, horizon, state_dim))
    dummy_t = jnp.ones((1, 1))
    dummy_cond = jnp.ones((1, cond_dim)) if cond_dim else None
    _ = model_handle.apply(params, dummy_X_t, dummy_t, cond=dummy_cond, deterministic=True)

    return model_handle, params


def count_parameters(params: nnx.State) -> int:
    """Count the number of parameters in a model."""
    return sum(int(x.size) for x in jax.tree_util.tree_leaves(params))


def params_to_bytes(params: nnx.State) -> bytes:
    """Serialize nnx parameter state to bytes for checkpointing."""
    pure = statelib.to_pure_dict(params)
    return serialization.to_bytes(pure)


def bytes_to_params(data: Union[bytes, jnp.ndarray], template: nnx.State) -> nnx.State:
    """Deserialize bytes into the nnx parameter State using a template."""
    pure_template = statelib.to_pure_dict(template)
    pure_params = serialization.from_bytes(pure_template, data)
    restored = copy.deepcopy(template)
    statelib.replace_by_pure_dict(restored, pure_params)
    return restored
