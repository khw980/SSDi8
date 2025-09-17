import math
import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64, 'BLOCK_SIZE_K':32}, num_warps=2, num_stages=4),
    ],
    key=['chunk_size', 'K'],
)
@triton.jit
def _int8_cb_bmm_kernel(
    # int8 inputs
    c_q_ptr, b_q_ptr,
    # ⬇️ 추가: 출력 포인터
    out_ptr,
    # per-group scales
    Sc_ptr, Sb_ptr, Scb_ptr,
    # optional seq mask
    seq_idx_ptr,
    # dims
    seqlen, chunk_size, K, ngroups,
    # strides
    sc_b, sc_s, sc_g, sc_k,
    sb_b, sb_s, sb_g, sb_k,
    so_b, so_c, so_g, so_m, so_n,   # out strides (B,C,G,M,N)
    sseq_b, sseq_s,
    # meta
    HAS_GROUPS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b  = tl.program_id(axis=1)
    pid_cg = tl.program_id(axis=2)            # (chunk, group) or just chunk
    pid_c  = pid_cg // ngroups if HAS_GROUPS else pid_cg
    pid_g  = pid_cg - pid_c * ngroups if HAS_GROUPS else 0

    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) %  num_pid_n

    # pointers base
    c_q_ptr += pid_b*sc_b + pid_c*chunk_size*sc_s + (pid_g*sc_g if HAS_GROUPS else 0)
    b_q_ptr += pid_b*sb_b + pid_c*chunk_size*sb_s + (pid_g*sb_g if HAS_GROUPS else 0)

    offs_m = pid_m*BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # in [0, chunk_size)
    offs_n = pid_n*BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    c_ptrs = c_q_ptr + (offs_m[:,None]*sc_s + offs_k[None,:]*sc_k)         # (M,K)
    b_ptrs = b_q_ptr + (offs_k[:,None]*sb_k + offs_n[None,:]*sb_s)         # (K,N)

    chunk_size_limit = tl.minimum(chunk_size, seqlen - pid_c*chunk_size)

    acc32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    # K loop: pure INT8×INT8→INT32
    for k0 in range(0, K, BLOCK_SIZE_K):
        mask_mk = (offs_m[:,None] < chunk_size_limit) & (offs_k[None,:] < (K - k0))
        mask_kn = (offs_k[:,None] < (K - k0)) & (offs_n[None,:] < chunk_size_limit)
        Cblk = tl.load(c_ptrs, mask=mask_mk, other=0).to(tl.int8)
        Bblk = tl.load(b_ptrs, mask=mask_kn, other=0).to(tl.int8)
        acc32 += tl.dot(Cblk, Bblk, out_dtype=tl.int32)
        c_ptrs += BLOCK_SIZE_K * sc_k
        b_ptrs += BLOCK_SIZE_K * sb_k

    # optional same-segment mask
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b*sseq_b + pid_c*chunk_size*sseq_s
        seq_m = tl.load(seq_idx_ptr + offs_m * sseq_s, mask=offs_m < chunk_size_limit, other=-1)
        seq_n = tl.load(seq_idx_ptr + offs_n * sseq_s, mask=offs_n < chunk_size_limit, other=-2)
        acc32 = tl.where(seq_m[:,None] == seq_n[None,:], acc32, 0)

    # requant int32 -> int8 using per-group scale
    Sc = tl.load(Sc_ptr + pid_g).to(tl.float32)
    Sb = tl.load(Sb_ptr + pid_g).to(tl.float32)
    Scb = tl.load(Scb_ptr + pid_g).to(tl.float32)   # target scale for CB_q
    req = (Sc * Sb) / Scb
    acc_fp = tl.cast(acc32, tl.float32) * req
    q = tl.where(acc_fp >= 0, tl.floor(acc_fp + 0.5), tl.ceil(acc_fp - 0.5))
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    q8 = tl.cast(q, tl.int8)

    # store CB_q: shape (B,C,G,M,N)
    # ✅ 포인터 베이스 이동은 += 로!
    out_ptr += pid_b*so_b + pid_c*so_c + (pid_g*so_g if HAS_GROUPS else 0)
    out_ptrs = out_ptr + (so_m*offs_m[:, None] + so_n*offs_n[None, :])

    # ✅ 패딩 경계 일관화
    mask_store = (offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < chunk_size_limit)
    tl.store(out_ptrs, q8, mask=mask_store)


def _quant_bmm_chunk_fwd(
    q_C, q_B, S_c_g, S_b_g, S_cb_g, chunk_size, seq_idx=None
):
    """
    q_C: int8 (B,S,G,K)   , q_B: int8 (B,S,G,K)
    S_c_g, S_b_g, S_cb_g: (G,) float32  (per-group scales)
    return: CB_q int8 (B, C, G, chunk, chunk)
    """
    Bsz, S, G, K = q_C.shape
    assert q_B.shape == (Bsz, S, G, K)
    C = (S + chunk_size - 1) // chunk_size

    CB_q = torch.empty((Bsz, C, G, chunk_size, chunk_size), device=q_C.device, dtype=torch.int8)

    grid = lambda META: (
        triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
        Bsz,
        C*G
    )
    with torch.cuda.device(q_C.device.index):
        _int8_cb_bmm_kernel[grid](
            q_C, q_B,CB_q,
            S_c_g, S_b_g, S_cb_g,
            seq_idx if seq_idx is not None else q_C,   # dummy ptr if None
            S, chunk_size, K, G,
            q_C.stride(0), q_C.stride(1), q_C.stride(2), q_C.stride(3),
            q_B.stride(0), q_B.stride(1), q_B.stride(2), q_B.stride(3),
            CB_q.stride(0), CB_q.stride(1), CB_q.stride(2), CB_q.stride(3), CB_q.stride(4),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0,0)),
            HAS_GROUPS=True,
            HAS_SEQ_IDX=(seq_idx is not None),
        )
    return CB_q


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'K', 'IS_CAUSAL'],
)
@triton.jit
def _quamba2_bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr, a_scale, b_ptr, b_scale, out_ptr, seq_idx_ptr,
    # Matrix dimensions
    seqlen, chunk_size, K, ngroups,
    stride_a_batch, stride_a_seqlen, stride_a_head, stride_ak,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_bk,
    stride_out_batch, stride_out_chunk, stride_out_head, stride_outm, stride_outn,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_scale_ptr = a_scale + pid_h
    b_scale_ptr = b_scale + pid_h
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_scale_ptr).to(dot_dtype) * tl.load(a_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K), other=0.0).to(dot_dtype)
        b = tl.load(b_scale_ptr).to(dot_dtype) * tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit), other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


def _quamba2_bmm_chunk_fwd(a, a_scale, b, b_scale, chunk_size, seq_idx=None, causal=False, output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = a.dim() == 4
    assert a_scale.is_cuda and b_scale.is_cuda
    if not has_groups:
        batch, seqlen, k = a.shape
        assert a_scale.numel() == 1 and b_scale.numel() == 1
    else:
        batch, seqlen, ngroups, k = a.shape
        assert a_scale.numel() == ngroups and b_scale.numel() == ngroups
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty((batch, nchunks, chunk_size, chunk_size) if not has_groups else (batch, nchunks, ngroups, chunk_size, chunk_size),
                      device=a.device, dtype=out_dtype)
    dot_dtype = (tl.bfloat16 if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16 or b.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                    batch, nchunks if not has_groups else nchunks * ngroups)
    with torch.cuda.device(a.device.index):
        _quamba2_bmm_chunk_fwd_kernel[grid](
            a, a_scale, b, b_scale, out, seq_idx,
            seqlen, chunk_size, k, ngroups if has_groups else 1,
            a.stride(0), a.stride(1), 0 if not has_groups else a.stride(2), a.stride(-1),
            b.stride(0), b.stride(1), 0 if not has_groups else b.stride(2), b.stride(-1),
            out.stride(0), out.stride(1), 0 if not has_groups else out.stride(2), out.stride(-2), out.stride(-1),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            causal,
            dot_dtype,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out