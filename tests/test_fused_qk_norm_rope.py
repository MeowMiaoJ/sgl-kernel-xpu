# Test for fused QK normalization and RoPE
# Adapted from the CUDA implementation in sglang

import math
import sys

import pytest
import sgl_kernel
import torch
import utils
from test_rope_utils import create_cos_sin_cache

precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}
device = utils.get_device()


def llama_rms_norm(x, w, eps=1e-6):
    """PyTorch reference implementation of RMS normalization."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * w.float()
    x = x.to(orig_dtype)
    return x


def apply_rotary_emb_native(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Native PyTorch rotary embedding implementation.
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, rotary_dim // 2]
        sin: [num_tokens, rotary_dim // 2]
        is_neox_style: Whether to use Neox-style or interleaved style
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        # Neox style: split in half along head dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        # Interleaved style: even and odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def compute_inv_freq_yarn(
    head_dim: int,
    rotary_dim: int,
    base: float,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
):
    """Compute inverse frequencies for YARN RoPE."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )

    if factor != 1.0:
        # YARN scaling
        dim_range = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)

        # Compute linear interpolation factor
        linear_func = (dim_range - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        ramp_func = torch.clamp(linear_func, 0.0, 1.0)

        inv_freq_extrapolation = inv_freq
        inv_freq_interpolation = inv_freq / factor

        inv_freq = (
            inv_freq_interpolation * (1.0 - ramp_func)
            + inv_freq_extrapolation * ramp_func
        )

    return inv_freq


def fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    base: float,
    is_neox: bool,
    position_ids: torch.Tensor,
    factor: float = 1.0,
    low: float = 1.0,
    high: float = 1.0,
    attention_factor: float = 1.0,
    rotary_dim: int = None,
) -> torch.Tensor:
    """
    Reference implementation in PyTorch for testing.

    Args:
        qkv: [num_tokens, (num_heads_q + num_heads_k + num_heads_v) * head_dim]
        Other args match the kernel interface
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Reshape QKV to separate Q, K, V
    qkv_reshaped = qkv.view(num_tokens, total_heads, head_dim)

    q = qkv_reshaped[:, :num_heads_q, :]
    k = qkv_reshaped[:, num_heads_q : num_heads_q + num_heads_k, :]
    v = qkv_reshaped[:, num_heads_q + num_heads_k :, :]

    # Apply RMSNorm to Q and K
    q_normalized = llama_rms_norm(q, q_weight, eps)
    k_normalized = llama_rms_norm(k, k_weight, eps)

    # Compute RoPE frequencies
    inv_freq = compute_inv_freq_yarn(head_dim, rotary_dim, base, factor, low, high)

    # Compute cos and sin for each position. Ensure both tensors are on the
    # same device to avoid cross-device ops (tests sometimes pass CPU tensors
    # as reference while inv_freq is constructed on `device`).
    positions = position_ids.to(torch.float32)
    inv_freq = inv_freq.to(positions.device)
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()

    # Apply attention factor
    cos = cos * attention_factor
    sin = sin * attention_factor

    # Apply RoPE to Q and K (only to rotary_dim portion)
    q_rot = q_normalized[..., :rotary_dim]
    q_pass = q_normalized[..., rotary_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_final = torch.cat([q_rot, q_pass], dim=-1)

    k_rot = k_normalized[..., :rotary_dim]
    k_pass = k_normalized[..., rotary_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_final = torch.cat([k_rot, k_pass], dim=-1)

    # Concatenate Q, K, V back together
    result = torch.cat([q_final, k_final, v], dim=1)
    result = result.view(num_tokens, total_heads * head_dim)

    return result


def fused_qk_norm_rope_with_cache_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for the cache-based fused QK norm + RoPE path."""
    head_dim = q.shape[-1]
    rope_dim = cos_sin_cache.shape[-1]
    positions = positions.flatten()
    flat_tokens = positions.numel()

    assert rope_dim % 2 == 0
    assert rope_dim <= head_dim

    cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
    cos = cos_cache[positions].to(q.dtype)
    sin = sin_cache[positions].to(q.dtype)

    q_view = q.reshape(flat_tokens, -1, head_dim)
    k_view = k.reshape(flat_tokens, -1, head_dim)

    q_norm = llama_rms_norm(q_view, q_weight, eps)
    k_norm = llama_rms_norm(k_view, k_weight, eps)

    q_rot = q_norm[..., :rope_dim]
    q_pass = q_norm[..., rope_dim:]
    q_rot = apply_rotary_emb_native(q_rot, cos, sin, is_neox)
    q_out = torch.cat((q_rot, q_pass), dim=-1).reshape(q.shape)

    k_rot = k_norm[..., :rope_dim]
    k_pass = k_norm[..., rope_dim:]
    k_rot = apply_rotary_emb_native(k_rot, cos, sin, is_neox)
    k_out = torch.cat((k_rot, k_pass), dim=-1).reshape(k.shape)

    return q_out, k_out


@pytest.mark.parametrize("num_tokens", [1, 7, 32, 128])
@pytest.mark.parametrize("num_heads_q", [8, 32])
@pytest.mark.parametrize("num_heads_k", [8])
@pytest.mark.parametrize("num_heads_v", [8])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_norm_rope_basic(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox, dtype
):
    """Test basic fused QK norm + RoPE without YARN."""
    torch.random.manual_seed(42)
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [32, 128])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_yarn(num_tokens, head_dim, is_neox, dtype):
    """Test fused QK norm + RoPE with YARN scaling."""
    torch.random.manual_seed(42)
    num_heads_q = 32
    num_heads_k = 8
    num_heads_v = 8
    eps = 1e-6
    base = 10000.0
    factor = 2.0  # YARN factor
    low = 8.0
    high = 1024.0
    attention_factor = 0.707  # sqrt(0.5)
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results - use slightly relaxed tolerance for YARN
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype] * 2, atol=precision[dtype] * 2
    )


@pytest.mark.parametrize("num_tokens", [64])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_norm_rope_partial_rotary(num_tokens, head_dim, rotary_dim, dtype):
    """Test with partial rotary dimensions (rotary_dim < head_dim)."""
    torch.random.manual_seed(42)
    num_heads_q = 16
    num_heads_k = 4
    num_heads_v = 4
    eps = 1e-6
    base = 10000.0
    is_neox = True
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors
    qkv = torch.randn(num_tokens, total_heads * head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference
    qkv_ref = qkv.clone().float()
    q_weight_ref = q_weight.clone().float()
    k_weight_ref = k_weight.clone().float()
    position_ids_ref = position_ids.clone()

    # Compute reference output
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(dtype)

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results
    torch.testing.assert_close(
        qkv, output_ref, rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "num_heads_q,num_heads_k,num_heads_v",
    [
        (2, 2, 2),
        (1, 1, 1),
        (4, 2, 2),
        (8, 4, 2),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("is_neox", [True, False])
def test_fused_qk_norm_rope_fp8_e4m3(
    num_tokens, num_heads_q, num_heads_k, num_heads_v, head_dim, is_neox
):
    """Test fused QK norm + RoPE with FP8_E4M3 dtype."""
    torch.random.manual_seed(42)
    dtype = torch.float8_e4m3fn
    eps = 1e-6
    base = 10000.0
    factor = 1.0
    low = 1.0
    high = 1.0
    attention_factor = 1.0
    rotary_dim = head_dim

    total_heads = num_heads_q + num_heads_k + num_heads_v

    # Create input tensors in float32 first, then convert to FP8
    qkv_f32 = torch.randn(
        num_tokens, total_heads * head_dim, dtype=torch.float32, device=device
    )
    # Clamp to FP8 representable range to avoid infinities/NaNs on conversion
    qkv_f32 = qkv_f32.clamp(-448.0, 448.0)
    qkv = qkv_f32.to(dtype)

    q_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    q_weight_f32 = q_weight_f32.clamp(-448.0, 448.0)
    q_weight = q_weight_f32.to(dtype)

    k_weight_f32 = torch.randn(head_dim, dtype=torch.float32, device=device)
    k_weight_f32 = k_weight_f32.clamp(-448.0, 448.0)
    k_weight = k_weight_f32.to(dtype)

    position_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)

    # Create a copy for reference from FP8-dequantized values
    qkv_ref = qkv.to(torch.float32).clone().cpu()
    q_weight_ref = q_weight.to(torch.float32).clone().cpu()
    k_weight_ref = k_weight.to(torch.float32).clone().cpu()
    position_ids_ref = position_ids.clone().cpu()

    # Compute reference output on CPU
    output_ref = fused_qk_norm_rope_reference(
        qkv_ref,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight_ref,
        k_weight_ref,
        base,
        is_neox,
        position_ids_ref,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    ).to(device)

    # Run kernel (in-place operation)
    sgl_kernel.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        is_neox,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
    )

    # Compare results - use relaxed tolerance for FP8
    # FP8 has limited precision, so we need higher tolerance
    torch.testing.assert_close(qkv.to(torch.float32), output_ref, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize(
    "use_4d,batch_size,seq_len,num_qo_heads,num_kv_heads,head_dim,rope_dim,is_neox,dtype,position_dtype",
    [
        (False, 3, None, 4, 2, 64, 32, False, torch.bfloat16, torch.int32),
        (False, 5, None, 8, 4, 128, 64, True, torch.float16, torch.int64),
        (True, 2, 4, 16, 4, 128, 128, False, torch.bfloat16, torch.int32),
        (True, 1, 8, 32, 8, 256, 128, True, torch.float16, torch.int64),
    ],
)
def test_fused_qk_norm_rope_with_cache(
    use_4d,
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rope_dim,
    is_neox,
    dtype,
    position_dtype,
):
    """Test fused QK norm + RoPE with a precomputed cos/sin cache."""
    torch.random.manual_seed(42)

    assert rope_dim <= head_dim

    if use_4d:
        assert seq_len is not None
        q = torch.randn(
            batch_size, seq_len, num_qo_heads, head_dim, dtype=dtype, device=device
        )
        k = torch.randn(
            batch_size, seq_len, num_kv_heads, head_dim, dtype=dtype, device=device
        )
        num_tokens = batch_size * seq_len
    else:
        q = torch.randn(batch_size, num_qo_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, num_kv_heads, head_dim, dtype=dtype, device=device)
        num_tokens = batch_size

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=position_dtype, device=device)
    cos_sin_cache = create_cos_sin_cache(rope_dim, max_position=num_tokens + 1)

    q_ref, k_ref = fused_qk_norm_rope_with_cache_reference(
        q.clone().float(),
        k.clone().float(),
        q_weight.clone().float(),
        k_weight.clone().float(),
        cos_sin_cache,
        positions,
        is_neox,
    )

    q_test = q.clone()
    k_test = k.clone()
    sgl_kernel.fused_inplace_qknorm_rope(
        q_test,
        k_test,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox,
    )

    torch.testing.assert_close(
        q_test, q_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )
    torch.testing.assert_close(
        k_test, k_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )


@pytest.mark.parametrize(
    "use_4d,batch_size,seq_len,num_qo_heads,num_kv_heads,head_dim,rope_dim,is_neox,dtype,position_dtype,last_dim_padding",
    [
        (False, 3, None, 4, 2, 64, 32, False, torch.bfloat16, torch.int32, 16),
        (True, 2, 4, 16, 4, 128, 128, True, torch.float16, torch.int64, 32),
    ],
)
def test_fused_qk_norm_rope_with_cache_last_dim_strided(
    use_4d,
    batch_size,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rope_dim,
    is_neox,
    dtype,
    position_dtype,
    last_dim_padding,
):
    """Test fused QK norm + RoPE with non-contiguous Q/K views."""
    torch.random.manual_seed(42)

    assert rope_dim <= head_dim

    if use_4d:
        assert seq_len is not None
        q_storage = torch.randn(
            batch_size,
            seq_len,
            num_qo_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        k_storage = torch.randn(
            batch_size,
            seq_len,
            num_kv_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        num_tokens = batch_size * seq_len
    else:
        q_storage = torch.randn(
            batch_size,
            num_qo_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        k_storage = torch.randn(
            batch_size,
            num_kv_heads,
            head_dim + last_dim_padding,
            dtype=dtype,
            device=device,
        )
        num_tokens = batch_size

    q = q_storage[..., :head_dim]
    k = k_storage[..., :head_dim]
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert not q.is_contiguous()
    assert not k.is_contiguous()

    q_weight = torch.randn(head_dim, dtype=dtype, device=device)
    k_weight = torch.randn(head_dim, dtype=dtype, device=device)
    positions = torch.arange(num_tokens, dtype=position_dtype, device=device)
    cos_sin_cache = create_cos_sin_cache(rope_dim, max_position=num_tokens + 1)

    q_ref, k_ref = fused_qk_norm_rope_with_cache_reference(
        q.clone().float(),
        k.clone().float(),
        q_weight.clone().float(),
        k_weight.clone().float(),
        cos_sin_cache,
        positions,
        is_neox,
    )

    sgl_kernel.fused_inplace_qknorm_rope(
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        is_neox,
    )

    torch.testing.assert_close(
        q, q_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )
    torch.testing.assert_close(
        k, k_ref.to(dtype), rtol=precision[dtype], atol=precision[dtype]
    )


def _ref_rmsnorm_self(x: torch.Tensor, eps: float) -> torch.Tensor:
    rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype)


def _ref_rope_interleaved(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, rope_dim: int
) -> torch.Tensor:
    out = x.clone()
    batch_size = x.size(0)
    head_dim = x.size(-1)
    nope_dim = head_dim - rope_dim

    for b in range(batch_size):
        pos = int(positions[b].item())
        freq = freqs_cis[pos]
        rope_part = out[b, ..., nope_dim:].float()
        pairs = rope_part.reshape(*rope_part.shape[:-1], rope_dim // 2, 2)
        x_real = pairs[..., 0]
        x_imag = pairs[..., 1]
        freq_pairs = freq.reshape(rope_dim // 2, 2)
        f_real = freq_pairs[:, 0]
        f_imag = freq_pairs[:, 1]
        rot_real = x_real * f_real - x_imag * f_imag
        rot_imag = x_real * f_imag + x_imag * f_real
        result = torch.stack([rot_real, rot_imag], dim=-1).reshape(rope_part.shape)
        out[b, ..., nope_dim:] = result.to(x.dtype)

    return out


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("head_dim", [128, 192])
def test_fused_q_norm_rope_reference(batch_size, num_heads, head_dim):
    """Test DeepSeek-V4 fused_q_norm_rope: Q-only RMSNorm (no weight) + RoPE
    against a freqs_cis-based interleaved (real, imag) reference."""
    torch.manual_seed(42)
    rope_dim = 64
    max_pos = 512
    eps = 1e-6

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    freqs_cis = torch.randn(max_pos, rope_dim, dtype=torch.float32, device=device)
    positions = torch.randint(
        0, max_pos, (batch_size,), dtype=torch.int32, device=device
    )

    q_output = sgl_kernel.fused_q_norm_rope(q_input, freqs_cis, positions, eps)

    normed = _ref_rmsnorm_self(q_input, eps)
    expected = _ref_rope_interleaved(normed, freqs_cis, positions, rope_dim)

    torch.testing.assert_close(q_output.float(), expected.float(), rtol=1e-2, atol=1e-2)


def test_fused_q_norm_rope_zero_batch():
    q_input = torch.empty(0, 8, 192, dtype=torch.bfloat16, device=device)
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device=device)
    positions = torch.empty(0, dtype=torch.int32, device=device)

    q_output = sgl_kernel.fused_q_norm_rope(q_input, freqs_cis, positions)
    assert q_output.shape == q_input.shape


def test_fused_q_norm_rope_preallocated_output():
    torch.manual_seed(42)
    batch_size, num_heads, head_dim = 4, 8, 192

    q_input = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    freqs_cis = torch.randn(512, 64, dtype=torch.float32, device=device)
    positions = torch.randint(0, 512, (batch_size,), dtype=torch.int32, device=device)
    q_output = torch.empty_like(q_input)

    result = sgl_kernel.fused_q_norm_rope(
        q_input,
        freqs_cis,
        positions,
        q_output=q_output,
    )

    assert result is q_output


# ---------------------------------------------------------------------------
# fused_k_norm_rope_flashmla
# ---------------------------------------------------------------------------
#
# This op has no standalone output tensor: it writes each token's
# normalized+roped row directly into a paged FlashMLA KV-cache buffer as
# 448 fp8-e4m3 "nope" bytes + 128 bf16 "rope" bytes, plus 7 UE8M0 scale
# bytes (+1 padding byte). To test it we dequantize the cache back out and
# compare against a PyTorch reference of the norm+rope math (allowing for
# fp8 quantization error on the nope part).

_KV_HEAD_DIM = 512
_KV_ROPE_DIM = 64
_KV_NOPE_DIM = _KV_HEAD_DIM - _KV_ROPE_DIM  # 448
_KV_NOPE_SUBGROUPS = 7
_KV_ELEMS_PER_SUBGROUP = _KV_NOPE_DIM // _KV_NOPE_SUBGROUPS  # 64
_KV_VALUE_BYTES = 576  # 448 fp8 (1B) + 64 bf16 (2B)
_KV_SCALE_BYTES = 8  # 7 UE8M0 scales + 1 padding byte


def _flashmla_page_bytes(page_size: int) -> int:
    return (
        ((_KV_VALUE_BYTES + _KV_SCALE_BYTES) * page_size + _KV_VALUE_BYTES - 1)
        // _KV_VALUE_BYTES
        * _KV_VALUE_BYTES
    )


def _dequant_flashmla_cache(
    kvcache: torch.Tensor, out_loc: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode fused_k_norm_rope_flashmla's cache layout back into
    (nope [T, 448] float32, rope [T, 64] float32)."""
    page_bits = int(math.log2(page_size))
    page_bytes = _flashmla_page_bytes(page_size)
    kvcache_cpu = kvcache.cpu()

    num_tokens = out_loc.shape[0]
    nope_out = torch.empty(num_tokens, _KV_NOPE_DIM, dtype=torch.float32)
    rope_out = torch.empty(num_tokens, _KV_ROPE_DIM, dtype=torch.float32)

    for i in range(num_tokens):
        loc = int(out_loc[i].item())
        page = loc >> page_bits
        offset = loc & ((1 << page_bits) - 1)
        page_base = page * page_bytes
        value_base = page_base + offset * _KV_VALUE_BYTES
        scale_base = page_base + _KV_VALUE_BYTES * page_size + offset * _KV_SCALE_BYTES

        value_bytes = kvcache_cpu[value_base : value_base + _KV_VALUE_BYTES]
        nope_bytes = value_bytes[:_KV_NOPE_DIM]
        rope_bytes = value_bytes[_KV_NOPE_DIM:]
        scale_bytes = kvcache_cpu[scale_base : scale_base + _KV_NOPE_SUBGROUPS]

        nope_fp8 = nope_bytes.contiguous().view(torch.float8_e4m3fn).float()
        for w in range(_KV_NOPE_SUBGROUPS):
            scale = 2.0 ** (int(scale_bytes[w].item()) - 127)
            lo, hi = w * _KV_ELEMS_PER_SUBGROUP, (w + 1) * _KV_ELEMS_PER_SUBGROUP
            nope_out[i, lo:hi] = nope_fp8[lo:hi] * scale

        rope_out[i] = rope_bytes.contiguous().view(torch.bfloat16).float()

    return nope_out, rope_out


def _ref_k_norm_rope_flashmla(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference: RMSNorm(kv, kv_weight) over all 512 dims, then interleaved
    (real, imag) RoPE on the tail 64 dims. Returns float32 [T, 512]."""
    normed = llama_rms_norm(kv, kv_weight, eps).float()
    return _ref_rope_interleaved(normed, freqs_cis, positions, _KV_ROPE_DIM)


@pytest.mark.parametrize("num_tokens,page_size", [(1, 8), (3, 4), (16, 8), (20, 8)])
def test_fused_k_norm_rope_flashmla_reference(num_tokens, page_size):
    """Test DeepSeek-V4 fused_k_norm_rope_flashmla: weighted RMSNorm + RoPE,
    encoded directly into a paged fp8(nope)+bf16(rope) FlashMLA cache."""
    torch.manual_seed(42)
    eps = 1e-6
    max_pos = 512

    kv = torch.randn(num_tokens, _KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn(_KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
    freqs_cis = torch.randn(max_pos, _KV_ROPE_DIM, dtype=torch.float32, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int32, device=device
    )
    out_loc = torch.arange(num_tokens, dtype=torch.int32, device=device)

    num_pages = num_tokens // page_size + 2
    kvcache = torch.zeros(
        num_pages * _flashmla_page_bytes(page_size), dtype=torch.uint8, device=device
    )

    sgl_kernel.fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_cis, positions, out_loc, kvcache, page_size, eps
    )

    expected = _ref_k_norm_rope_flashmla(
        kv.clone().float(), kv_weight.clone().float(), freqs_cis, positions, eps
    )
    expected_nope = expected[:, :_KV_NOPE_DIM]
    expected_rope = expected[:, _KV_NOPE_DIM:]

    actual_nope, actual_rope = _dequant_flashmla_cache(
        kvcache, out_loc.cpu(), page_size
    )

    # fp8 e4m3 has ~3 mantissa bits, so use a generous relative tolerance for
    # the quantized nope part; rope is stored as plain bf16.
    torch.testing.assert_close(actual_nope, expected_nope.cpu(), rtol=0.15, atol=0.1)
    torch.testing.assert_close(actual_rope, expected_rope.cpu(), rtol=1e-2, atol=1e-2)


def test_fused_k_norm_rope_flashmla_zero_tokens():
    kv = torch.empty(0, _KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
    kv_weight = torch.randn(_KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
    freqs_cis = torch.randn(512, _KV_ROPE_DIM, dtype=torch.float32, device=device)
    positions = torch.empty(0, dtype=torch.int32, device=device)
    out_loc = torch.empty(0, dtype=torch.int32, device=device)
    kvcache = torch.zeros(_flashmla_page_bytes(8), dtype=torch.uint8, device=device)

    # Should be a no-op and must not raise.
    sgl_kernel.fused_k_norm_rope_flashmla(
        kv, kv_weight, freqs_cis, positions, out_loc, kvcache, 8, 1e-6
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
