# ============================================================
# Fused Triton kernel for quantized chunk state forward
#  - X/B quantization (with scales)
#  - LUT (exp(dA_last - dA) * dt) on-the-fly
#  - int8 x int8 GEMM -> int32
#  - row/col dequant (alpha x B_scale) to fp16/fp32
# Shapes used:
#   X:        (B, S, H, P)           (S = C*K)
#   B:        (B, S, G, N)
#   alpha:    (B, H, C, P)           (row scale for X)
#   bscale:   (B, G, C, N)           (col scale for B)
#   dA, dt:   (B, H, C, K)
#   mask(b,c,k) optional: (B, C, K)  (last segment mask per chunk)
#   states:   (B, C, H, P, N)
# ============================================================

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _quant_chunk_state_fwd_kernel_fused(
    # 입력
    x_ptr, b_ptr,                    # x: FP,  b: INT8  (항상)
    alpha_ptr,                       # (B,H,C,P) row scale for X
    bscale_ptr,                      # (B,G,C,N) col scale for B  (출력 requant용)
    dA_ptr, dt_ptr,                  # (B,H,C,K)
    mask_ptr,                        # (B,C,K) or dummy if USE_MASK=0
    states_ptr,                      # (B,C,H,P,N) int8 출력
    out_qscale_ptr,                  # (H,P)  S_out
    # sizes
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # strides...
    sx_b, sx_s, sx_h, sx_p,
    sb_b, sb_s, sb_g, sb_n,
    ss_b, ss_c, ss_h, ss_p, ss_n,
    sa_b, sa_h, sa_c, sa_p,
    sbs_b, sbs_g, sbs_c,
    sos_h, sos_p,
    sDA_b, sDA_h, sDA_c, sDA_k,
    sdt_b, sdt_h, sdt_c, sdt_k,
    sm_b, sm_c, sm_k,
    # meta
    STATES_FP32: tl.constexpr,
    USE_MASK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c  = pid_bc // batch
    pid_b  = pid_bc - pid_c * batch
    pid_h  = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) %  num_pid_n
    gid = pid_h // nheads_ngroups_ratio

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # P
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # N
    offs_k = tl.arange(0, BLOCK_SIZE_K)                         # K

    tl.static_assert(BLOCK_SIZE_K % 4 == 0)
    tl.multiple_of(offs_k, 16); tl.multiple_of(offs_m, 16); tl.multiple_of(offs_n, 16)

    x_base = x_ptr + pid_b * sx_b + pid_h * sx_h + (pid_c * CHUNK_SIZE) * sx_s
    b_base = b_ptr + pid_b * sb_b + gid   * sb_g + (pid_c * CHUNK_SIZE) * sb_s

    x_ptrs = x_base + (offs_m[:, None] * sx_p + offs_k[None, :] * sx_s)
    b_ptrs = b_base + (offs_n[None, :] * sb_n + offs_k[:, None] * sb_s)

    chunk_size_limit = tl.minimum(CHUNK_SIZE, seqlen - pid_c * CHUNK_SIZE)

    # row/col scales
    a_base  = alpha_ptr  + pid_b * sa_b + pid_h * sa_h + pid_c * sa_c
    a_ptrs  = a_base + offs_m * sa_p
    alpha_m = tl.load(a_ptrs, mask=offs_m < hdim, other=1.0, cache_modifier=".ca")
    inv_alpha_m = 1.0 / alpha_m

    bs_base = bscale_ptr + pid_b * sbs_b + gid * sbs_g + pid_c * sbs_c
    bscale_g = tl.load(bs_base).to(tl.float32)  # scalar

    dA_last_ix = pid_b * sDA_b + pid_h * sDA_h + pid_c * sDA_c + (CHUNK_SIZE - 1) * sDA_k
    dA_last = tl.load(dA_ptr + dA_last_ix).to(tl.float32)

    acc_i32 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k0 in tl.static_range(0, CHUNK_SIZE, BLOCK_SIZE_K):
        k_eff = chunk_size_limit - k0
        mask_k = offs_k < k_eff

        # X: FP → (inv_alpha * LUT) → int8
        x_fp = tl.load(x_ptrs,
                       mask=(offs_m[:, None] < hdim) & mask_k[None, :],
                       other=0.0, cache_modifier=".ca").to(tl.float32)
        dA_ptrs = (pid_b * sDA_b + pid_h * sDA_h + pid_c * sDA_c) + (k0 + offs_k) * sDA_k
        dt_ptrs = (pid_b * sdt_b + pid_h * sdt_h + pid_c * sdt_c) + (k0 + offs_k) * sdt_k
        dA_k = tl.load(dA_ptr + dA_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptr + dt_ptrs, mask=mask_k, other=0.0).to(tl.float32)
        lut_k = tl.exp(dA_last - dA_k) * dt_k
        if USE_MASK:
            m_ptrs = pid_b * sm_b + pid_c * sm_c + (k0 + offs_k) * sm_k
            m_k = tl.load(mask_ptr + m_ptrs, mask=mask_k, other=0.0)
            lut_k = lut_k * m_k

        x_fp *= inv_alpha_m[:, None]
        x_fp *= lut_k[None, :]
        x_q = tl.where(x_fp >= 0, tl.floor(x_fp + 0.5), tl.ceil(x_fp - 0.5))
        x_q = tl.maximum(tl.minimum(x_q, 127), -127).to(tl.int8)

        # B: 항상 int8로 로드
        b_q = tl.load(b_ptrs,
                      mask=mask_k[:, None] & (offs_n[None, :] < dstate),
                      other=0).to(tl.int8)

        acc_i32 += tl.dot(x_q, b_q)

        x_ptrs += BLOCK_SIZE_K * sx_s
        b_ptrs += BLOCK_SIZE_K * sb_s

    # requant & store int8 states
    oq_ptrs = out_qscale_ptr + pid_h * sos_h + offs_m * sos_p
    S_out = tl.load(oq_ptrs, mask=(offs_m < hdim), other=1.0).to(tl.float32)  # (BM,)

    # (BM,1) * (1,BN) 이지만 per-G 스칼라라서 열 독립: 쉽게 (BM,BN) 브로드캐스트됨
    req_scale = (alpha_m / S_out)[:, None] * bscale_g  # bscale_g: scalar
    acc_fp = tl.cast(acc_i32, tl.float32) * req_scale

    q = tl.where(acc_fp >= 0, tl.floor(acc_fp + 0.5), tl.ceil(acc_fp - 0.5))
    q = tl.maximum(tl.minimum(q, 127), -127).to(tl.int8)

    s_base = states_ptr + pid_b * ss_b + pid_c * ss_c + pid_h * ss_h
    s_ptrs = s_base + (offs_m[:, None] * ss_p + offs_n[None, :] * ss_n)
    tl.store(s_ptrs, q, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate))





def _quant_chunk_state_fwd(
        q_B,  q_x, 
        dt, dA_cumsum, mm_dtype=torch.float16,  # mm_dtype kept for API parity; not used now
        seq_idx=None, states=None, states_in_fp32=True,
        x_row_scale_chp: torch.Tensor | None = None,   # (H,P)
        B_scale_g: torch.Tensor | None = None,   # (G,N),
        state_chunkscan_scale: torch.Tensor = None,  # (H,P
    ):
    """
    Fused path: everything happens in the Triton kernel.
    Inputs:
      q_x:  (B,S,H,P) FP16/BF16/FP32
      q_B:  (B,S,G,N) FP16/BF16/FP32
      dt, dA_cumsum: (B,H,C,K)
      x_row_scale_chp: (H,P)   -> expanded to (B,H,C,P)
      B_row_scale_cgn: (G,N)   -> expanded to (B,G,C,N)
      seq_idx (optional): (B,S) -> last-segment mask per chunk
    Output:
      states: (B,C,H,P,N) FP16/FP32  (dtype via states_in_fp32)
    """
    import time
    from contextlib import contextmanager
    dev = q_B.device if q_B.is_cuda else None

    @contextmanager
    def _time_block(label: str, device=None, sync=True):
        use_cuda = device is not None and torch.cuda.is_available()
        if use_cuda:
            if sync: torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            try: yield
            finally:
                end.record()
                if sync: torch.cuda.synchronize(device)
                ms = start.elapsed_time(end); print(f"[TIME] {label:<32} {ms:.3f} ms")
        else:
            t0 = time.perf_counter()
            try: yield
            finally:
                t1 = time.perf_counter(); print(f"[TIME] {label:<32} {(t1 - t0)*1000:.3f} ms")

    # --- shapes / checks ---
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape

    assert q_x.is_cuda and q_B.is_cuda and dt.is_cuda and dA_cumsum.is_cuda
    assert nheads % ngroups == 0
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert x_row_scale_chp is not None and B_scale_g is not None
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

    states = torch.empty((batch, nchunks, nheads, headdim, dstate),
                         device=q_x.device, dtype=torch.int8)

    out_qscale = state_chunkscan_scale.to(q_x.device, torch.float32)  # (H,P) = S_state_hp


    # --- pad S to C*K if needed ---
    S_total = nchunks * chunk_size
    if seqlen != S_total:
            q_x_pad = torch.zeros((batch, S_total, nheads, headdim), device=q_x.device, dtype=q_x.dtype)
            q_x_pad[:, :seqlen] = q_x
            q_x = q_x_pad
            seqlen_kernel = S_total
    else:
            seqlen_kernel = seqlen

    if q_B.shape[1] != seqlen_kernel:
            q_B_pad = torch.zeros((batch, seqlen_kernel, ngroups, dstate), device=q_B.device, dtype=q_B.dtype)
            q_B_pad[:, :seqlen] = q_B
            q_B = q_B_pad

    # --- alpha_x (B,H,C,P) ---
    alpha_x_bhcp = x_row_scale_chp.to(q_x.device)           # (H,P)
    alpha_x_bhcp = alpha_x_bhcp.unsqueeze(0).unsqueeze(2)   # (1,H,1,P)
    alpha_x_bhcp = alpha_x_bhcp.expand(batch, nheads, nchunks, headdim)  # (B,H,C,P)

    # --- B_scales (B,G,C,N) ---

        # B_scale_g: (G,)  ← 위에서 quantize_int8_per_group가 반환한 두 번째 값 사용 권장
    B_scale_g = B_scale_g.to(q_B.device)  # (G,)

    B_scales_bgc = B_scale_g.view(1, ngroups, 1) \
                                .expand(batch, ngroups, nchunks)  # (B,G,C)
    B_scales = B_scales_bgc

    # --- optional last-segment mask (B,C,K) ---
    if seq_idx is not None:
            seg = seq_idx.view(batch, nchunks, chunk_size)
            mask_bck = seg.eq(seg[..., -1:].expand_as(seg)).to(dt.dtype)
    else:
            mask_bck = None

    # --- launch fused kernel ---

    grid = lambda META: (
            triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
            batch * nchunks,
            nheads
        )
    with torch.cuda.device(q_x.device.index):
            _quant_chunk_state_fwd_kernel_fused[grid](
                # ptrs
                q_x, q_B,                      # q_B는 int8 (per-G quant) 버전
                alpha_x_bhcp, B_scales,        # B_scales: (B,G,C)
                dA_cumsum, dt,
                (mask_bck if mask_bck is not None else q_B),
                states, out_qscale,
                # sizes
                headdim, dstate, chunk_size,
                batch, seqlen_kernel, nheads // ngroups,
                # strides: X (B,S,H,P)
                q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
                # strides: B (B,S,G,N)
                q_B.stride(0), q_B.stride(1), q_B.stride(2), q_B.stride(3),
                # strides: states (B,C,H,P,N)
                states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
                # strides: alpha (B,H,C,P)
                alpha_x_bhcp.stride(0), alpha_x_bhcp.stride(1), alpha_x_bhcp.stride(2), alpha_x_bhcp.stride(3),
                # strides: bscale (B,G,C)  ← 변경
                B_scales.stride(0), B_scales.stride(1), B_scales.stride(2),
                # out_qscale (H,P)
                out_qscale.stride(0), out_qscale.stride(1),
                # strides: dA, dt (B,H,C,K)
                dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2), dA_cumsum.stride(3),
                dt.stride(0),        dt.stride(1),        dt.stride(2),        dt.stride(3),
                # strides: mask (B,C,K)
                *(mask_bck.stride() if mask_bck is not None else (0, 0, 0)),
                # meta
                STATES_FP32=False,
                USE_MASK=(mask_bck is not None),
                CHUNK_SIZE=chunk_size,
            )

    return states
# ##########################################################################################
#     try:
#         from pathlib import Path
#         import matplotlib.pyplot as plt
#         import numpy as np
#         from matplotlib import colors as mcolors

#         # === 스타일 전용 헬퍼: 색맵/두께만 통일 ===
#         def _styled_bar3d(ax, X, Y, Z, *, cmap="coolwarm", alpha=0.9, width=0.5):
#             """
#             X, Y: np.meshgrid 출력 (각각 shape=(L, P) 등)
#             Z   : 위와 같은 shape의 값 행렬 (표시 높이)
#             """
#             vals = np.abs(Z).ravel()
#             cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
#             vmin = float(vals.min())
#             vmax = float(max(vals.max(), 1e-8))
#             norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#             facecol = cmap_fn(norm(vals))

#             Xr = X.ravel() + 0.5 - width/2
#             Yr = Y.ravel() + 0.5 - width/2
#             Z0 = np.zeros_like(vals)

#             ax.bar3d(
#                 Xr, Yr, Z0,
#                 width, width, vals,
#                 color=facecol,
#                 shade=False,
#                 edgecolor="none",
#                 linewidth=0.0,
#                 alpha=alpha,
#             )

#         def plot_tensor_3d(name: str,
#                         tensor: torch.Tensor,
#                         save_dir="mamba_plots_in_state",
#                         cmap="coolwarm",
#                         alpha=0.9,
#                         width=0.5,
#                         *,
#                         b_idx: int = 0,
#                         h_idx: int = 0):
#             """
#             name== "q_B"                : (B, L, G(orH), N) -> B=b_idx, G/H=0 slice -> (L, N)
#             name== "q_X with scale"     : (B, L, H, P)      -> B=b_idx, H=h_idx slice -> (L, P)
#             name== "X_scaled_pxl_by_c"  : (B, C, K, H, P)   -> B=b_idx, H=h_idx, for each c -> (K, P)
#             """
#             save_dir = Path(save_dir)
#             save_dir.mkdir(parents=True, exist_ok=True)

#             if name == "q_B":
#                 mat = tensor[b_idx, :, 0].detach().float().cpu().numpy()  # (L, N)
#                 X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
#                 Z = mat.T  # (N, L)

#                 fig = plt.figure(figsize=(8, 4))
#                 ax = fig.add_subplot(111, projection="3d")
#                 _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
#                 ax.set(title=name, xlabel="L", ylabel="dim", zlabel="val")
#                 fig.savefig(save_dir / f"{name}.png", dpi=160); plt.close(fig)

#             elif name == "q_X":
#                 mat = tensor[b_idx, :, h_idx].detach().float().cpu().numpy()  # (L, P)
#                 X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
#                 Z = mat.T  # (P, L)

#                 fig = plt.figure(figsize=(8, 4))
#                 ax = fig.add_subplot(111, projection="3d")
#                 _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
#                 ax.set(title=name + f" (b={b_idx}, h={h_idx})", xlabel="L", ylabel="P", zlabel="|val|")
#                 fig.savefig(save_dir / f"{name}_b{b_idx}_h{h_idx}.png", dpi=160); plt.close(fig)

#             elif name == "X_scaled_pxl_by_c":
#                 # tensor: (B, C, K, H, P)
#                 B, C, K, H, P = tensor.shape
#                 for c in range(C):
#                     # (K, P) slice for given chunk c, fixed batch/head
#                     mat = tensor[b_idx, c, :, h_idx, :].detach().float().cpu().numpy()  # (K, P)
#                     X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))  # (K, P)
#                     Z = mat.T  # (P, K)로 전치해서 Y축이 P가 되도록 시각화

#                     fig = plt.figure(figsize=(8, 4))
#                     ax = fig.add_subplot(111, projection="3d")
#                     _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
#                     ax.set(title=f"{name} (b={b_idx}, h={h_idx}, c={c})",
#                         xlabel="K (chunk L)", ylabel="P (headdim)", zlabel="|val|")
#                     fig.savefig(save_dir / f"{name}_b{b_idx}_h{h_idx}_c{c}.png", dpi=160)
#                     plt.close(fig)
#             else:
#                 return  # 지원 안 하는 이름은 무시

#         # === 실제 호출 ===
#         plot_tensor_3d("q_B", q_B, save_dir="mamba_plots_in_state")

#         # 기존 q_x 대신, 스케일 적용 전/후를 각기 보고 싶으면 둘 다 호출 가능
#         # (1) q_X (원본 q_x; H 슬라이스)
#         plot_tensor_3d("q_X", q_x.view(batch, seqlen_kernel, nheads, headdim),
#                     save_dir="mamba_plots_in_state", b_idx=0, h_idx=0)

#         # (2) X_scaled: 각 chunk c마다 (K × P)로 저장
#         plot_tensor_3d("X_scaled_pxl_by_c", X_scaled,
#                     save_dir="mamba_plots_in_state", b_idx=0, h_idx=0)

#     except Exception as e:
#         print("[plot] error:", e)
#    return states






@triton.jit
def _quamba2_chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, x_scales_ptr, x_head_group_range_ptr, x_dim_group_range_ptr,
    b_ptr, b_scale, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # group quant paramters
    nhead_groups: tl.constexpr,
    ndim_groups: tl.constexpr,
    # Meta-parameters
    USE_FLOA32_MM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_scale_ptr = b_scale + (pid_h // nheads_ngroups_ratio)
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # load x_head_group_range: [n_ssd_groups, n_head_groups]
    x_head_group_range_ptr = x_head_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups
    x_head_group_range = tl.load(x_head_group_range_ptr + tl.arange(0, nhead_groups))
    x_head_gidx = tl.sum(tl.where(pid_h % nheads_ngroups_ratio >= x_head_group_range, 1, 0))
    # load x_dim_group_range: [n_ssd_groups, n_head_groups, n_dim_groups]
    x_dim_group_range_ptr = x_dim_group_range_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_dim_group_range_ptr = x_dim_group_range_ptr + x_head_gidx*ndim_groups
    x_dim_group_range = tl.load(x_dim_group_range_ptr + tl.arange(0, ndim_groups))
    # load x_scales
    x_dim_gidx = tl.sum(tl.where(offs_m[:, None] >= x_dim_group_range[None, :], 1, 0), axis=-1)
    x_scales_ptr = x_scales_ptr + (pid_h // nheads_ngroups_ratio)*nhead_groups*ndim_groups
    x_scales_ptrs = x_scales_ptr + (x_head_gidx*ndim_groups + x_dim_gidx)
    x_scales = tl.load(x_scales_ptrs)

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    if USE_FLOA32_MM:
        mm_dtype = tl.float32
    else:
        mm_dtype = tl.float16
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # scaling factors are float32 
        x = x_scales[:, None] * tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_scale_ptr) * tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(mm_dtype)
        x = x.to(mm_dtype)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _quamba2_chunk_state_fwd(
        q_B, B_scale, q_x, x_scales, x_head_group_range, x_dim_group_range,
        dt, dA_cumsum, mm_dtype=torch.float16,
        seq_idx=None, states=None, states_in_fp32=True
    ):
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_dim_groups]
    assert q_x.is_cuda
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups * ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32
    assert nheads % ngroups == 0

    assert B_scale.is_cuda
    assert B_scale.numel() == ngroups
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert mm_dtype is torch.float16 or mm_dtype is torch.float32
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else torch.float16
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=q_x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(q_x.device.index):
        _quamba2_chunk_state_fwd_kernel[grid](
            q_x, x_scales, x_head_group_range, x_dim_group_range,
            q_B, B_scale, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            q_x.stride(0), q_x.stride(1), q_x.stride(2), q_x.stride(3),
            q_B.stride(0), q_B.stride(1), q_B.stride(2), q_B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            nhead_groups=nhead_groups,
            ndim_groups=ndim_groups,
            USE_FLOA32_MM = mm_dtype is torch.float32,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states




# --------- 공통: 안전한 반올림(트라이던트 Round 이슈 회피용) ----------

@triton.jit
def _round_half_away_from_zero(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))

@triton.jit
def quantize_x_bcKHP_kernel(
    x_ptr, sc_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
    C: tl.constexpr, K: tl.constexpr,
    sx_b, sx_s, sx_h, sx_p,
    ss_b, ss_c, ss_k, ss_h, ss_p,
    so_b, so_s, so_h, so_p,
    BLOCK_P: tl.constexpr,
):
    pid_p  = tl.program_id(0)           # x-dim: tile along P
    pid_y  = tl.program_id(1)           # y-dim
    pid_z  = tl.program_id(2)           # z-dim

    # 합쳐서 (B*S*H) 선형 인덱스 만들기
    nprog_y = tl.num_programs(1)
    pid_lin = pid_y + pid_z * nprog_y
    total   = B * S * H
    # 과잉 프로그램이면 일찍 종료
    if pid_lin >= total:
        return

    # (b, s, h) 복원
    h = pid_lin % H
    tmp = pid_lin // H
    s = tmp % S
    b = tmp // S

    # P 타일
    p_offsets = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = p_offsets < P

    # S -> (C,K) 분해
    c = s // K
    k = s - c * K

    # 오프셋
    x_ix  = b*sx_b + s*sx_s + h*sx_h + p_offsets*sx_p
    sc_ix = b*ss_b + c*ss_c + k*ss_k + h*ss_h + p_offsets*ss_p
    o_ix  = b*so_b + s*so_s + h*so_h + p_offsets*so_p

    x  = tl.load(x_ptr  + x_ix,  mask=mask_p, other=0.0)
    sc = tl.load(sc_ptr + sc_ix, mask=mask_p, other=1.0)

    # mul → round → clamp → cast(int8)
    y = x * sc
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    y = tl.minimum(y, 127)
    y = tl.maximum(y, -127)
    y_i8 = y.to(tl.int8)            # ← tl.astype 대신 .to

    tl.store(out_ptr + o_ix, y_i8, mask=mask_p)

def quantize_x_triton_bcKHP(x_scaled, inv_alpha_bcKHP, chunk_size):
    assert x_scaled.is_cuda and inv_alpha_bcKHP.is_cuda
    if x_scaled.dim() == 5:  # (B,C,K,H,P)
        B,C,K,H,P = x_scaled.shape
        assert K == chunk_size
        S = C * K
        xs = x_scaled.reshape(B, S, H, P)
    else:                     # (B,S,H,P)
        B,S,H,P = x_scaled.shape
        assert S % chunk_size == 0
        C = S // chunk_size; K = chunk_size
        xs = x_scaled

    assert inv_alpha_bcKHP.shape == (B, C, K, H, P)

    out = torch.empty_like(xs, dtype=torch.int8)

    BLOCK_P = 128
    gx = triton.cdiv(P, BLOCK_P)
    total = B * S * H
    # CUDA 한계(65535)를 넘지 않도록 y/z로 분할
    max_y = 65535
    gy = min(total, max_y)
    gz = (total + gy - 1) // gy

    quantize_x_bcKHP_kernel[(gx, gy, gz)](
        xs, inv_alpha_bcKHP, out,
        B, S, H, P, C, K,
        *xs.stride(),
        *inv_alpha_bcKHP.stride(),
        *out.stride(),
        BLOCK_P=BLOCK_P,
        num_warps=4, num_stages=2
    )
    return out
# --------- B 경로: (B,G,C,K,N) × (B,G,C,N) ----------
@triton.jit
def quantize_b_kernel(
    b_ptr, scale_ptr, out_ptr,
    B: tl.constexpr, G: tl.constexpr, C: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    sb_b, sb_g, sb_c, sb_k, sb_n,
    ss_b, ss_g, ss_c, ss_n,
    so_b, so_g, so_c, so_k, so_n,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)

    nprog_y = tl.num_programs(1)
    pid_lin = pid_y + pid_z * nprog_y
    total   = B * G * C * K
    if pid_lin >= total:
        return

    # (b,g,c,k) 복원
    k = pid_lin % K
    tmp = pid_lin // K
    c = tmp % C
    tmp //= C
    g = tmp % G
    b = tmp // G

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    b_ix  = b*sb_b + g*sb_g + c*sb_c + k*sb_k + n_offsets*sb_n
    sc_ix = b*ss_b + g*ss_g + c*ss_c + n_offsets*ss_n
    o_ix  = b*so_b + g*so_g + c*so_c + k*so_k + n_offsets*so_n

    x  = tl.load(b_ptr + b_ix, mask=mask_n, other=0.0)
    sc = tl.load(scale_ptr + sc_ix, mask=mask_n, other=1.0)

    y = x * sc
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    y = tl.minimum(y, 127)
    y = tl.maximum(y, -127)
    y_i8 = y.to(tl.int8)

    tl.store(out_ptr + o_ix, y_i8, mask=mask_n)

def quantize_b_triton(B_fp: torch.Tensor, inv_B: torch.Tensor) -> torch.Tensor:
    """
    B_fp: (B,G,C,K,N) in fp16/bf16
    inv_B: (B,G,C,N)  in fp16/bf16  (K축으로 브로드캐스트)
    returns int8 tensor (B,G,C,K,N)
    """
    assert B_fp.is_cuda and inv_B.is_cuda
    B,G,C,K,N = B_fp.shape
    assert inv_B.shape == (B,G,C,N)

    out = torch.empty_like(B_fp, dtype=torch.int8)

    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N), B * G * C * K)

    quantize_b_kernel[(grid)](
        B_fp, inv_B, out,
        B, G, C, K, N,
        *B_fp.stride(), *inv_B.stride(), *out.stride(),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )
    return out

@triton.jit
def quantize_x_dyn_kernel(
    x_ptr,            # (B,S,H,P)  fp16/bf16
    dA_ptr, dt_ptr,   # (B,H,C,K)  fp16/bf16/float32
    alpha_ptr,        # (B,C,1,H,P) fp16/bf16  (HP 고정 스케일, K축 브로드캐스트)
    mask_ptr,         # (B,C,K) 또는 더미 (USE_MASK=0이면 무시)
    out_ptr,          # (B,S,H,P)  int8

    # sizes
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr, P: tl.constexpr,
    C: tl.constexpr, K: tl.constexpr,

    # strides
    sx_b, sx_s, sx_h, sx_p,
    sDA_b, sDA_h, sDA_c, sDA_k,
    sdt_b, sdt_h, sdt_c, sdt_k,
    sa_b, sa_c, sa_1, sa_h, sa_p,
    sm_b, sm_c, sm_k,    # mask strides (USE_MASK==0이면 0 전달해도 OK)
    so_b, so_s, so_h, so_p,

    # meta
    USE_MASK: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # 3D grid → (b,s,h) 복원
    pid_p = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    nprog_y = tl.num_programs(1)
    pid_lin = pid_y + pid_z * nprog_y
    total = B * S * H
    if pid_lin >= total:
        return

    h = pid_lin % H
    tmp = pid_lin // H
    s = tmp % S
    b = tmp // S

    # s → (c,k)
    c = s // K
    k = s - c * K

    p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    mask_p = p < P

    # X load
    x_ix = b*sx_b + s*sx_s + h*sx_h + p*sx_p
    x = tl.load(x_ptr + x_ix, mask=mask_p, other=0.0, cache_modifier=".ca")

    # scale_lut = exp(dA_last - dA[k]) * dt[k]   (스칼라)
    dA_last_ix = b*sDA_b + h*sDA_h + c*sDA_c + (K-1)*sDA_k
    dA_ix      = b*sDA_b + h*sDA_h + c*sDA_c + k*sDA_k
    dt_ix      = b*sdt_b + h*sdt_h + c*sdt_c + k*sdt_k

    dA_last = tl.load(dA_ptr + dA_last_ix).to(tl.float32)
    dA_val  = tl.load(dA_ptr + dA_ix).to(tl.float32)
    dt_val  = tl.load(dt_ptr + dt_ix).to(tl.float32)
    lut = tl.exp(dA_last - dA_val) * dt_val  # scalar fp32

    # 선택적 마스킹 (마지막 segment 외 0)
    if USE_MASK:
        m_ix = b*sm_b + c*sm_c + k*sm_k
        m = tl.load(mask_ptr + m_ix).to(tl.float32)   # 0 or 1
        lut = lut * m

    # inv_alpha[P] 로드 (벡터)
    al_ix = b*sa_b + c*sa_c + 0*sa_1 + h*sa_h + p*sa_p
    alpha = tl.load(alpha_ptr + al_ix, mask=mask_p, other=1.0, cache_modifier=".ca")
    inv_alpha = 1.0 / alpha

    # mul → round → clamp → int8
    sc = lut.to(alpha.dtype) * inv_alpha
    y = x * sc
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    y = tl.maximum(tl.minimum(y, 127), -127)
    y_i8 = y.to(tl.int8)

    out_ix = b*so_b + s*so_s + h*so_h + p*so_p
    tl.store(out_ptr + out_ix, y_i8, mask=mask_p)



def quantize_x_dyn(X_fp_or_qx: torch.Tensor,
                   dA_cumsum: torch.Tensor,  # (B,H,C,K)
                   dt: torch.Tensor,         # (B,H,C,K)
                   alpha_bc1: torch.Tensor,  # (B,C,1,H,P)
                   chunk_size: int,
                   seq_idx: torch.Tensor | None = None  # (B,S) or None
                  ) -> torch.Tensor:
    """
    입력: X는 (B,C,K,H,P) 또는 (B,S,H,P).  S = C*K
    출력: (B,S,H,P) int8
    """
    assert X_fp_or_qx.is_cuda and dA_cumsum.is_cuda and dt.is_cuda and alpha_bc1.is_cuda
    # X 정규화 → (B,S,H,P)
    if X_fp_or_qx.dim() == 5:
        B, C, K, H, P = X_fp_or_qx.shape
        assert K == chunk_size
        S = C * K
        xs = X_fp_or_qx.reshape(B, S, H, P)
    else:
        B, S, H, P = X_fp_or_qx.shape
        assert S % chunk_size == 0
        C, K = S // chunk_size, chunk_size
        xs = X_fp_or_qx

    # shapes
    assert dA_cumsum.shape == (B, H, C, K) and dt.shape == (B, H, C, K)
    assert alpha_bc1.shape == (B, C, 1, H, P)

    # 선택적 마스크 (B,C,K)
    if seq_idx is not None:
        assert seq_idx.shape == (B, S)
        seg = seq_idx.view(B, C, K)
        mask_bck = (seg == seg[..., -1:].expand_as(seg)).to(dt.dtype)
    else:
        mask_bck = None

    out = torch.empty((B, S, H, P), device=xs.device, dtype=torch.int8)

    # grid
    BLOCK_P = 64 if P % 64 == 0 else 128
    gx = triton.cdiv(P, BLOCK_P)
    total = B * S * H
    gy = min(total, 65535)
    gz = (total + gy - 1) // gy

    # 호출
    quantize_x_dyn_kernel[(gx, gy, gz)](
        xs, dA_cumsum, dt, alpha_bc1,
        (mask_bck if mask_bck is not None else xs),  # dummy라도 포인터 필요
        out,
        # sizes
        B, S, H, P, C, K,
        # strides
        *xs.stride(),
        *dA_cumsum.stride(), *dt.stride(),
        *alpha_bc1.stride(),
        *(mask_bck.stride() if mask_bck is not None else (0, 0, 0)),
        *out.stride(),
        # meta
        USE_MASK=(mask_bck is not None),
        BLOCK_P=BLOCK_P,
        num_warps=4, num_stages=2
    )
    return out



