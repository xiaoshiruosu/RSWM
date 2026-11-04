import os
import warnings

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ImportError, OSError):
    FLASH_ATTN_2_AVAILABLE = False

__all__ = [
    'flash_attention',
    'attention',
    'attention_backend_status',
]

_BACKEND_ALIASES = {
    'auto': 'auto',
    'flash_attn': 'flash_attn',
    'flash-attn': 'flash_attn',
    'flash_attn_2': 'flash_attn',
    'fa2': 'flash_attn',
    'sdpa': 'sdpa',
}
_LAST_BACKEND = None
_LAST_FALLBACK_REASON = None


def _requested_backend():
    raw = os.environ.get('RSWM_ATTENTION_BACKEND', 'auto').strip().lower()
    if raw not in _BACKEND_ALIASES:
        warnings.warn(
            f'Unknown RSWM_ATTENTION_BACKEND={raw!r}; using auto. '
            'Expected auto, flash_attn, or sdpa.')
        return 'auto'
    return _BACKEND_ALIASES[raw]


def attention_backend_status():
    """Return a lightweight, JSON-serializable attention capability report."""
    return {
        'requested': _requested_backend(),
        'flash_attn_2_available': bool(FLASH_ATTN_2_AVAILABLE),
        'flash_attn_3_available': bool(FLASH_ATTN_3_AVAILABLE),
        'last_backend': _LAST_BACKEND,
        'last_fallback_reason': _LAST_FALLBACK_REASON,
    }


def _select_backend(q, fa_version=None):
    requested = _requested_backend()
    if requested == 'sdpa' or q.device.type != 'cuda':
        return 'sdpa', None

    major = torch.cuda.get_device_capability(q.device)[0]
    # FA3 is a Hopper path. Never select it on A800/A100 (sm80), even if a
    # stray flash_attn_interface package can be imported in that environment.
    if major >= 9 and FLASH_ATTN_3_AVAILABLE and fa_version in (None, 3):
        return 'flash_attn', 3
    if major >= 8 and FLASH_ATTN_2_AVAILABLE:
        return 'flash_attn', 2

    if requested == 'flash_attn':
        warnings.warn(
            'flash_attn was requested but no compatible kernel is available; '
            'falling back to PyTorch SDPA.')
    return 'sdpa', None


def _sdpa_attention(q, k, v, q_lens, k_lens, dropout_p, causal, dtype,
                    softmax_scale=None, q_scale=None,
                    window_size=(-1, -1)):
    """Correctness-first SDPA fallback for BSHD inputs.

    Variable-length batches are evaluated sample-by-sample so fallback does
    not allocate a potentially enormous dense padding mask.
    """
    out_dtype = q.dtype
    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q = q * q_scale
    if window_size != (-1, -1):
        warnings.warn(
            'PyTorch SDPA fallback does not implement FlashAttention local '
            'windows; using global attention for this fallback call.')

    b, _, lq, _ = q.shape
    lk = k.shape[2]

    def lengths(value, default):
        if value is None:
            return [default] * b
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        return [max(0, min(default, int(item))) for item in value]

    q_sizes = lengths(q_lens, lq)
    k_sizes = lengths(k_lens, lk)
    sdpa_kwargs = dict(
        attn_mask=None, is_causal=causal, dropout_p=dropout_p)
    if softmax_scale is not None:
        sdpa_kwargs['scale'] = softmax_scale

    if all(size == lq for size in q_sizes) and all(
            size == lk for size in k_sizes):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, **sdpa_kwargs)
    else:
        warnings.warn(
            'Variable-length FlashAttention call fell back to per-sample '
            'PyTorch SDPA; this is correct but slower.')
        chunks = []
        for index, (q_size, k_size) in enumerate(zip(q_sizes, k_sizes)):
            if q_size == 0 or k_size == 0:
                chunk = q.new_zeros((1, q.shape[1], lq, v.shape[-1]))
            else:
                valid = torch.nn.functional.scaled_dot_product_attention(
                    q[index:index + 1, :, :q_size],
                    k[index:index + 1, :, :k_size],
                    v[index:index + 1, :, :k_size], **sdpa_kwargs)
                chunk = torch.nn.functional.pad(valid, (0, 0, 0, lq - q_size))
            chunks.append(chunk)
        out = torch.cat(chunks, dim=0)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def _flash_attention_impl(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """Public attention entry used by Wan blocks, with hardware dispatch."""
    global _LAST_BACKEND, _LAST_FALLBACK_REASON
    backend, selected_fa_version = _select_backend(q, version)
    if backend == 'flash_attn':
        try:
            result = _flash_attention_impl(
                q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
                dropout_p=dropout_p, softmax_scale=softmax_scale,
                q_scale=q_scale, causal=causal, window_size=window_size,
                deterministic=deterministic, dtype=dtype,
                version=selected_fa_version)
            _LAST_BACKEND = f'flash_attn_{selected_fa_version}'
            _LAST_FALLBACK_REASON = None
            return result
        except (RuntimeError, AssertionError, OSError, TypeError) as exc:
            _LAST_FALLBACK_REASON = f'{type(exc).__name__}: {exc}'
            warnings.warn(
                f'FlashAttention failed ({type(exc).__name__}: {exc}); '
                'falling back to PyTorch SDPA for this call.')
    else:
        _LAST_FALLBACK_REASON = None
    _LAST_BACKEND = 'sdpa'
    return _sdpa_attention(
        q, k, v, q_lens, k_lens, dropout_p, causal, dtype,
        softmax_scale=softmax_scale, q_scale=q_scale,
        window_size=window_size)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    return flash_attention(
        q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
        dropout_p=dropout_p, softmax_scale=softmax_scale,
        q_scale=q_scale, causal=causal, window_size=window_size,
        deterministic=deterministic, dtype=dtype, version=fa_version)
