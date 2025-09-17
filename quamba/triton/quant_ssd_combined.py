import torch
from einops import rearrange, repeat

from quamba.triton.quant_chunk_cumsum import _quant_chunk_cumsum_fwd
from quamba.triton.quant_state_passing import _quant_state_passing_fwd
from quamba.triton.quant_chunk_state import _quant_chunk_state_fwd, _quamba2_chunk_state_fwd
from quamba.triton.quant_chunk_scan import _quant_chunk_scan_fwd, _quamba2_chunk_scan_fwd
from quamba.triton.quant_bmm_chunk import _quant_bmm_chunk_fwd, _quamba2_bmm_chunk_fwd
from quamba.triton.quant_ssm_states import _quant_quant_ssm_states, _quamba2_quant_ssm_states

def _quant_mamba_chunk_scan_combined_fwd(
        q_x, x_scale, q_dt, dt_scale, q_A_log, A_log_scale,
        q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16,    x_row_scale_chp=None,  ori_x_row_scale_chp=None ,
        C_chunkscan_scale=None,cb_chunkscan_scale=None,state_chunkscan_scale=None,
        B_row_scale_cgn=None, comp_calib=False
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape

    assert x_scale.is_cuda
    assert x_scale.numel() == 1
    assert B_scale.is_cuda
    assert B_scale.numel() == 1
    assert C_scale.is_cuda
    assert C_scale.numel() == 1

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    #assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    #assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    #assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    import time
    from contextlib import contextmanager

    def _fmt_bytes(x: int) -> str:
            return f"{x/1024/1024:.1f}MB"

    @contextmanager
    def _time_block(label: str, device=None, sync: bool=True, mem: bool=True, reset_peak: bool=False):
            """
            - 시간 측정: CUDA면 cudaEvent, 아니면 perf_counter
            - 메모리 측정(mem=True):
                * 시작/끝 alloc/reserved
                * 블록 내 peak alloc/reserved (reset_peak=True면 블록 시작 전에 피크 리셋)
            """
            use_cuda = (device is not None) and torch.cuda.is_available()

            # ---- pre: time + mem ----
            if use_cuda:
                if sync: torch.cuda.synchronize(device)
                start_ev = torch.cuda.Event(enable_timing=True); end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()

                if mem:
                    if reset_peak:
                        torch.cuda.reset_peak_memory_stats(device)
                    alloc0   = torch.cuda.memory_allocated(device)
                    reserv0  = torch.cuda.memory_reserved(device)
            else:
                t0 = time.perf_counter()

            try:
                yield
            finally:
                # ---- post: time + mem ----
                if use_cuda:
                    end_ev.record()
                    if sync: torch.cuda.synchronize(device)
                    ms = start_ev.elapsed_time(end_ev)

                    if mem:
                        alloc1   = torch.cuda.memory_allocated(device)
                        reserv1  = torch.cuda.memory_reserved(device)
                        peak_alloc  = torch.cuda.max_memory_allocated(device)
                        peak_reserv = torch.cuda.max_memory_reserved(device)

                        print(
                            f"[TIME] {label:<24} {ms:.3f} ms\n"
                            f"[MEM ] {label:<24} "
                            f"alloc {_fmt_bytes(alloc0)} → {_fmt_bytes(alloc1)} (Δ {_fmt_bytes(alloc1-alloc0)}) | "
                            f"reserved {_fmt_bytes(reserv0)} → {_fmt_bytes(reserv1)} (Δ {_fmt_bytes(reserv1-reserv0)}) | "
                            f"peak alloc {_fmt_bytes(peak_alloc)}, peak reserved {_fmt_bytes(peak_reserv)}"
                        )
                    else:
                        print(f"[TIME] {label:<24} {ms:.3f} ms")
                else:
                    t1 = time.perf_counter()
                    print(f"[TIME] {label:<24} {(t1 - t0)*1000:.3f} ms")
                    # CPU 메모리도 보고 싶으면 psutil 사용 (옵션)
                    # try:
                    #     import psutil, os
                    #     rss = psutil.Process(os.getpid()).memory_info().rss
                    #     print(f"[MEM ] {label:<24} RSS {_fmt_bytes(rss)}")
                    # except Exception:
                    #     pass

    dev = q_x.device if q_x.is_cuda else None
    # q_B, B_scale_g = quantize_int8_per_group_triton(q_B, B_row_scale_cgn, group_dim=2)
    if not comp_calib :
        q_B, B_scale_g = quantize_int8_per_group_triton(q_B, B_row_scale_cgn, group_dim=2)
    else :
        q_B, B_scale_g = quantize_int8_per_group_torch(q_B, B_row_scale_cgn, group_dim=2)
        
    if not comp_calib :
        q_C, C_scale_g = quantize_int8_per_group_triton(q_C, C_chunkscan_scale, group_dim=2)
    else :
        q_C, C_scale_g = quantize_int8_per_group_torch(q_C, C_chunkscan_scale, group_dim=2)

    # q_C, C_scale_g = quantize_int8_per_group_triton(q_C, C_chunkscan_scale, group_dim=2)


    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quant_chunk_state_fwd(
            q_B, q_x,
            dt, dA_cumsum,
            mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True,
            x_row_scale_chp=x_row_scale_chp, B_scale_g=B_scale_g,
            state_chunkscan_scale=state_chunkscan_scale,
        )
                

    states, final_states = _quant_state_passing_fwd(
                                    rearrange(states, "... p n -> ... (p n)"),
                                    dA_cumsum[:, :, :, -1],
                                    initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                        if initial_states is not None else None,
                                    seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                                )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]

    CB = _quant_bmm_chunk_fwd(q_C,q_B,C_scale_g,B_scale_g, cb_chunkscan_scale,chunk_size, seq_idx=seq_idx, )

    out, out_x = _quant_chunk_scan_fwd(
            CB,
            q_x, x_scale,
            dt, dA_cumsum,
            q_C, C_scale,
            states,
            q_D=q_D, 
            D_scale=D_scale,
            q_z=q_z, 
            z_scale=z_scale,
            seq_idx=seq_idx,
            mm_dtype=torch.float16,
            C_chunkscan_scale=C_scale_g,
            cb_chunkscan_scale=cb_chunkscan_scale,
            state_chunkscan_scale=state_chunkscan_scale,
            ori_x_row_scale_chp=ori_x_row_scale_chp,
        )

    # print(states.shape)
    # try:
    #     from pathlib import Path
    #     import matplotlib.pyplot as plt
    #     import numpy as np
    #     from matplotlib import colors as mcolors

    #     # === 스타일 전용 헬퍼: 색맵/두께만 통일 ===
    #     def _styled_bar3d(ax, X, Y, Z, *, cmap="coolwarm", alpha=0.9, width=0.5):
    #         """
    #         X, Y: np.meshgrid 출력 (각각 shape=(D, L) 혹은 (P, L) 등)
    #         Z   : 위와 같은 shape의 값 행렬 (표시 높이)
    #         """
    #         # 값은 abs로 통일 (원래 _bar3d와 동일)
    #         vals = np.abs(Z).ravel()
    #         # colormap 매핑
    #         cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    #         vmin = float(vals.min())
    #         vmax = float(max(vals.max(), 1e-8))
    #         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #         facecol = cmap_fn(norm(vals))

    #         # 막대 위치/크기 (두께=width) ― 기존 형식 유지
    #         Xr = X.ravel() + 0.5 - width/2
    #         Yr = Y.ravel() + 0.5 - width/2
    #         Z0 = np.zeros_like(vals)

    #         ax.bar3d(
    #             Xr, Yr, Z0,
    #             width, width, vals,
    #             color=facecol,
    #             shade=False,
    #             edgecolor="none",
    #             linewidth=0.0,
    #             alpha=alpha,
    #         )

    #     def plot_tensor_3d(name: str, tensor: torch.Tensor, save_dir="mamba_plots", cmap="coolwarm", alpha=0.9, width=0.5):
    #         save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    #         if name in ["q_B", "q_C"]:        # (B,L,G(orH),N) → B=0,H/G=0 slice → (L,N)
    #             mat = tensor[0, :, 0].detach().float().cpu().numpy()  # (L,N)
    #             X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    #             Z = mat.T  # (N,L)로 뒤집어 시각화

    #             fig = plt.figure(figsize=(8,4))
    #             ax = fig.add_subplot(111, projection="3d")
    #             _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
    #             ax.set(title=name, xlabel="L", ylabel="dim", zlabel="val")
    #             fig.savefig(save_dir / f"{name}.png", dpi=160); plt.close(fig)

    #         elif name == "states":           # (B,C,H,P,N) → B=0,H=0 → loop C개 (P,N)
    #             for ci in range(tensor.shape[1]):
    #                 mat = tensor[0, ci, 0].detach().float().cpu().numpy()   # (P,N)
    #                 X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    #                 Z = mat.T

    #                 fig = plt.figure(figsize=(8,4))
    #                 ax = fig.add_subplot(111, projection="3d")
    #                 _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
    #                 ax.set(title=f"states_c{ci}", xlabel="P", ylabel="N", zlabel="val")
    #                 fig.savefig(save_dir / f"states_c{ci}.png", dpi=160); plt.close(fig)

    #         elif name == "q_X":              # (B,L,H,P) → B=0,H=0 slice → (L,P)
    #             mat = tensor[0, :, 0].detach().float().cpu().numpy()        # (L,P)
    #             X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    #             Z = mat.T

    #             fig = plt.figure(figsize=(8,4))
    #             ax = fig.add_subplot(111, projection="3d")
    #             _styled_bar3d(ax, X, Y, Z, cmap=cmap, alpha=alpha, width=width)
    #             ax.set(title=name, xlabel="L", ylabel="dim", zlabel="val")
    #             fig.savefig(save_dir / f"{name}.png", dpi=160); plt.close(fig)

    #         else:
    #             return  # 지원 안 하는 이름은 무시

    #     # 실제 호출
    #     # plot_tensor_3d("q_B", q_B, save_dir="mamba_plots")
    #     # plot_tensor_3d("q_C", q_C, save_dir="mamba_plots")
    #     # plot_tensor_3d("states", states, save_dir="mamba_plots")
    #     # plot_tensor_3d("q_X", q_x, save_dir="mamba_plots")

    # # === 3D bar plots: dA/dt series (B=0, H=0) ===
    # from pathlib import Path
    # from datetime import datetime
    # import os
    # import matplotlib.pyplot as plt
    # import numpy as np

    # save_dir = Path("mamba_plots_BC")
    # save_dir.mkdir(parents=True, exist_ok=True)

    # # 런 식별자 (환경변수 RUN_TAG 우선, 없으면 타임스탬프-ms)
    # RUN_TAG = os.getenv("RUN_TAG", datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3])

    # def _save(fig, stem: str):
    #     """stem: '01_dA', '03_dA_minus_dAprime_c05' 등"""
    #     fig.savefig(save_dir / f"{RUN_TAG}_{stem}.png", dpi=160)
    #     plt.close(fig)

    # # 슬라이스: (B=0, H=0)
    # dA_bh_t = dA_cumsum[0, 0]   # (C, L) torch
    # dt_bh_t = dt[0, 0]          # (C, L) torch

    # # to numpy
    # dA_bh = dA_bh_t.detach().float().cpu().numpy()  # (C, L)
    # dt_bh = dt_bh_t.detach().float().cpu().numpy()  # (C, L)
    # C, L = dA_bh.shape

    # # helper: 같은 bar3d 스타일
    # def _styled_bar3d(ax, X, Y, Z, *, cmap="coolwarm", alpha=0.9, width=0.5):
    #     from matplotlib import colors as mcolors
    #     vals = np.abs(Z).ravel()
    #     cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    #     vmin = float(vals.min())
    #     vmax = float(max(vals.max(), 1e-8))
    #     norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    #     facecol = cmap_fn(norm(vals))
    #     Xr = X.ravel() + 0.5 - width/2
    #     Yr = Y.ravel() + 0.5 - width/2
    #     Z0 = np.zeros_like(vals)
    #     ax.bar3d(Xr, Yr, Z0, width, width, vals,
    #             color=facecol, shade=False, edgecolor="none",
    #             linewidth=0.0, alpha=alpha)



    # CB_t = CB[0]  # (C, G, L, L)
    # C, G, L, _ = CB_t.shape

    # # ---- Per-group scale (Scb). 없으면 1.0 사용 ----
    # try:
    #     Scb_all = cb_chunkscan_scale.detach().float().cpu().numpy().reshape(G)
    # except Exception:
    #     Scb_all = np.ones((G,), dtype=np.float32)

    # # ---- 마지막 청크 유효 길이(커널의 chunk_size_limit 반영) ----
    # global_seqlen = int(q_x.shape[1]) if 'q_x' in globals() else None
    # chunk_valid_lens = [L] * C
    # if global_seqlen is not None:
    #     for ci in range(C):
    #         start = ci * L
    #         chunk_valid_lens[ci] = max(0, min(L, global_seqlen - start))

    # # ---- causal mask (l >= l′) ----
    # tri_mask = (np.arange(L)[:, None] >= np.arange(L)[None, :]).astype(np.float32)

    # # ---- 설정: 모든 그룹 그릴지 여부 ----
    # PLOT_ALL_GROUPS = False
    # groups = range(G) if PLOT_ALL_GROUPS else [0]

    # for gi in groups:
    #     Scb = float(Scb_all[gi])
    #     for ci in range(C):
    #         # int8 → fp dequant
    #         cb_i8 = CB_t[ci, gi].detach().cpu().numpy().astype(np.int8)        # (L, L)
    #         cb_fp = cb_i8.astype(np.float32) * Scb

    #         # 마지막 청크 유효 길이 외부 0 처리
    #         vlen = chunk_valid_lens[ci]
    #         if vlen < L:
    #             cb_fp[vlen:, :] = 0.0
    #             cb_fp[:, vlen:] = 0.0

    #         # 07) raw CB
    #         X, Y = np.meshgrid(np.arange(L), np.arange(L))
    #         Z = cb_fp.T
    #         fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    #         _styled_bar3d(ax, X, Y, Z)
    #         ax.set(title=f"CB raw (B=0,c={ci},g={gi})", xlabel="l", ylabel="l′", zlabel="val")
    #         _save(fig, f"07_cb_raw_c{ci:02d}_g{gi:02d}")

    #         # 08) causal-masked CB (커널의 IS_CAUSAL 적용 버전)
    #         cb_causal = cb_fp * tri_mask
    #         if vlen < L:
    #             cb_causal[vlen:, :] = 0.0
    #             cb_causal[:, vlen:] = 0.0
    #         Z = cb_causal.T
    #         fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection="3d")
    #         _styled_bar3d(ax, X, Y, Z)
    #         ax.set(title=f"CB causal (B=0,c={ci},g={gi})", xlabel="l", ylabel="l′", zlabel="val")
    #         _save(fig, f"08_cb_causal_c{ci:02d}_g{gi:02d}")

    # print(f"[plot] saved ({RUN_TAG}): 07_cb_raw_c*_g*, 08_cb_causal_c*_g*")
    # # 1) dA[:, l]  → (C, L)
    # mat = dA_bh  # (C, L)
    # X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))  # (C, L)
    # Z = mat.T  # (L, C)로 표시
    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_subplot(111, projection="3d")
    # _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    # ax.set(title="dA (B=0,H=0)", xlabel="c", ylabel="l", zlabel="val")
    # _save(fig, "01_dA")

    # # 2) exp(dA[:, l])  → (C, L)
    # mat = np.exp(dA_bh)
    # X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    # Z = mat.T
    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_subplot(111, projection="3d")
    # _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    # ax.set(title="exp(dA) (B=0,H=0)", xlabel="c", ylabel="l", zlabel="val")
    # _save(fig, "02_exp_dA")

    # # 3) dA[:, l] - dA[:, l′]  → (C, L, L), c별 저장
    # with torch.no_grad():
    #     diff_t = dA_bh_t[:, :, None] - dA_bh_t[:, None, :]  # (C, L, L)
    # diff = diff_t.detach().float().cpu().numpy()
    # for ci in range(C):
    #     mat = diff[ci]  # (L, L)
    #     X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))  # (L, L′)
    #     Z = mat.T
    #     fig = plt.figure(figsize=(8, 4))
    #     ax = fig.add_subplot(111, projection="3d")
    #     _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    #     ax.set(title=f"dA(l) - dA(l′) (B=0,H=0,c={ci})", xlabel="l", ylabel="l′", zlabel="val")
    #     _save(fig, f"03_dA_minus_dAprime_c{ci:02d}")

    # # 4) dt  → (C, L)
    # mat = dt_bh
    # X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    # Z = mat.T
    # fig = plt.figure(figsize=(8, 4))
    # ax = fig.add_subplot(111, projection="3d")
    # _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    # ax.set(title="dt (B=0,H=0)", xlabel="c", ylabel="l", zlabel="val")
    # _save(fig, "04_dt")

    # # 5) for bc dA = exp(dA[:, l] - dA[:, l′])  → (C, L, L)
    # with torch.no_grad():
    #     bc_dA_t = torch.exp(dA_bh_t[:, :, None] - dA_bh_t[:, None, :])  # (C, L, L)
    # bc_dA = bc_dA_t.detach().float().cpu().numpy()
    # _, L, Lp = bc_dA.shape
    # for ci in range(C):
    #     mat = bc_dA[ci]  # (L, L)
    #     X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))  # (L, L′)
    #     Z = mat.T
    #     fig = plt.figure(figsize=(8, 4))
    #     ax = fig.add_subplot(111, projection="3d")
    #     _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    #     ax.set(title=f"for bc dA (B=0,H=0,c={ci})", xlabel="l", ylabel="l′", zlabel="val")
    #     _save(fig, f"05_for_bc_dA_c{ci:02d}")

    # # 6) for bc dAdt = (for bc dA) ⊙ dt(l′)  → (C, L, L)
    # bc_dAdt = bc_dA * dt_bh[:, None, :]
    # for ci in range(C):
    #     mat = bc_dAdt[ci]  # (L, L)
    #     X, Y = np.meshgrid(np.arange(mat.shape[0]), np.arange(mat.shape[1]))
    #     Z = mat.T
    #     fig = plt.figure(figsize=(8, 4))
    #     ax = fig.add_subplot(111, projection="3d")
    #     _styled_bar3d(ax, X, Y, Z, cmap="coolwarm", alpha=0.9, width=0.5)
    #     ax.set(title=f"for bc dAdt (B=0,H=0,c={ci})", xlabel="l", ylabel="l′", zlabel="val")
    #     _save(fig, f"06_for_bc_dAdt_c{ci:02d}")

    # print(f"[plot] saved set ({RUN_TAG}): 01_dA, 02_exp_dA, 03_dA_minus_dAprime_c*, 04_dt, 05_for_bc_dA_c*, 06_for_bc_dAdt_c*")




        # ======= PLOT: B=1, H=1 고정 후, 지정된 축 조합으로 per-col 통계 시각화 =======
    # try:
    #     import os
    #     import numpy as np
    #     import matplotlib
    #     matplotlib.use("Agg")  # 서버/터미널에서 저장만
    #     import matplotlib.pyplot as plt
    #     from pathlib import Path

    #     # -------- 공통 헬퍼: per-column (열) 통계 플롯 --------
    #     def _plot_col_stats(W2d: torch.Tensor, name="W",
    #                         save_dir="./mamba_plots",
    #                         y_lim=None, dpi=140,
    #                         colors=("forestgreen", "goldenrod", "maroon")):
    #         """
    #         W2d: (rows, cols). 각 열에 대해 행 방향으로 통계를 냄.
    #           - median(녹색), 99th(황금색), max(적갈색)
    #         """
    #         W = W2d.detach().float().cpu()
    #         if W.ndim != 2:
    #             W = W.reshape(W.shape[0], -1)

    #         med  = W.median(dim=0).values
    #         p99  = torch.quantile(W, 0.99, dim=0)
    #         vmax = W.max(dim=0).values
    #         xs   = np.arange(W.shape[1])

    #         fig, ax = plt.subplots(figsize=(9, 3), dpi=dpi)
    #         ax.fill_between(xs, 0,   med,  color=colors[0], alpha=.8, label="median")
    #         ax.fill_between(xs, med, p99,  color=colors[1], alpha=.8, label="p99")
    #         ax.fill_between(xs, p99, vmax, color=colors[2], alpha=.8, label="max")
    #         ax.set(xlabel="col index", ylabel="value", title=f"{name} — per-col stats")
    #         if y_lim is not None:
    #             ax.set_ylim(*y_lim)
    #         ax.margins(x=0)
    #         ax.legend(fontsize=8, framealpha=.9)
    #         fig.tight_layout()

    #         Path(save_dir).mkdir(parents=True, exist_ok=True)
    #         fname = Path(save_dir) / f"{name}.png"
    #         fig.savefig(fname)
    #         plt.close(fig)
    #         print(f"[plot] saved → {fname}")

    #     # -------- 축 고정/재배열 유틸 --------
    #     def _fix_B_and_H_or_G(t: torch.Tensor, batch: int, nheads: int, ngroups: int):
    #         """B=0, H=0(가능하면), 아니면 G=0으로 슬라이스"""
    #         if not torch.is_tensor(t): return None
    #         idx = [slice(None)] * t.ndim

    #         # B 고정
    #         for i, s in enumerate(t.shape):
    #             if s == batch:
    #                 idx[i] = 0
    #                 break

    #         # H 우선, 없으면 G
    #         fixed_head = False
    #         for i, s in enumerate(t.shape):
    #             if s == nheads and isinstance(idx[i], slice):
    #                 idx[i] = 0
    #                 fixed_head = True
    #                 break
    #         if not fixed_head:
    #             for i, s in enumerate(t.shape):
    #                 if s == ngroups and isinstance(idx[i], slice):
    #                     idx[i] = 0
    #                     break

    #         return t[tuple(idx)]

    #     PLOT_DIR = os.getenv("MAMBA_PLOT_DIR", "./mamba_plots")

    #     # =========================================================
    #     # C: shape (B, L, H(or G), N)
    #     #   1) B=1, H=1 고정 후, "L 에 대해 N 시각화"  → (rows=N, cols=L)
    #     #   2) B=1, H=1 고정 후, "N 에 대해 L 시각화"  → (rows=L, cols=N)
    #     # =========================================================
    #     C_fixed = _fix_B_and_H_or_G(C, batch=batch, nheads=nheads, ngroups=ngroups)

    #     def _as_LN(t: torch.Tensor, seqlen: int, dstate: int):
    #         """
    #         t → (L, N)으로 맞춤.
    #         - 3D면 L/N 축 찾아서 (L,N,*) → 첫 extra 축 평균 또는 첫 슬라이스
    #         - 2D면 (L,N) or (N,L) 판별해서 필요시 전치
    #         - 기타는 적당히 펴서 반환
    #         """
    #         if not torch.is_tensor(t):
    #             return None
    #         x = t.detach()

    #         # squeeze 차원 1들(혹시 남아있으면)
    #         while x.ndim > 2 and 1 in x.shape:
    #             x = x.squeeze()

    #         if x.ndim == 3:
    #             shape = x.shape
    #             axL = next((i for i,s in enumerate(shape) if s == seqlen), None)
    #             axN = next((i for i,s in enumerate(shape) if s == dstate and i != axL), None)
    #             if axL is None or axN is None:
    #                 # L/N 식별 실패 → 남은 축 평균해서 2D로 축소
    #                 x = x.float().mean(dim=0)
    #             else:
    #                 # (L, N, extra) 순서로 재배열
    #                 keep = [axL, axN]
    #                 extra = [i for i in range(3) if i not in keep]
    #                 x = x.permute(axL, axN, *extra).contiguous().float()
    #                 # extra 처리: 평균으로 2D화
    #                 if x.ndim == 3:
    #                     x = x.mean(dim=2)
    #         if x.ndim == 2:
    #             if x.shape == (seqlen, dstate):
    #                 return x
    #             elif x.shape == (dstate, seqlen):
    #                 return x.T
    #             else:
    #                 # 모양이 애매하면 첫 축을 L로 간주
    #                 return x
    #         if x.ndim == 1:
    #             return x.unsqueeze(1)
    #         return None

    #     C_LN = _as_LN(C_fixed, seqlen, dstate)
    #     if C_LN is not None:
    #         # 2) N에 대해 L 시각화: rows=L, cols=N (각 N의 시간분포) → 통계를 L 방향으로 집계
    #         _plot_col_stats(C_LN, name="C_B0H0__N_over_L",
    #                         save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))
    #         # 1) L에 대해 N 시각화: rows=N, cols=L (각 L의 차원분포) → 통계를 N 방향으로 집계
    #         _plot_col_stats(C_LN.T, name="C_B0H0__L_over_N",
    #                         save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))
    #     else:
    #         print("[plot] C: could not coerce to (L,N); skipped.")
    #     # =========================================================
    #     # states: shape (B, c, H, P, N)
    #     #   각 c마다
    #     #   1) "P에 대해서 N 시각화" → (rows=N, cols=P)
    #     #   2) "N에 대해서 P 시각화" → (rows=P, cols=N)
    #     # =========================================================
    #     if states is not None and states.ndim == 5:
    #         # B=0, H=0 고정
    #         # 찾아두기: c, P, N 축
    #         Bc, Cc, Hc, Pc, Nc = states.shape  # 예상 형태
    #         states_fixed = states[0, :, 0, :, :] if (Bc==batch and Hc==nheads) else _fix_B_and_H_or_G(states, batch, nheads, ngroups)
    #         # states_fixed 예상: (c, P, N)
    #         if states_fixed is not None and states_fixed.ndim == 3:
    #             c_len = states_fixed.shape[0]
    #             for ci in range(c_len):
    #                 S = states_fixed[ci]  # (P, N) (아닌 경우도 전치로 맞춤)

    #                 # P, N 축 정렬
    #                 # P=headdim, N=dstate 로 크기 매칭
    #                 axP = 0 if S.shape[0] == headdim else (1 if S.shape[1] == headdim else None)
    #                 axN = 0 if S.shape[0] == dstate else (1 if S.shape[1] == dstate else None)

    #                 if axP is None or axN is None:
    #                     # 모양 불분명하면 일단 (P,N)로 가정
    #                     S_PN = S if S.shape[0] == headdim else S.t()
    #                 else:
    #                     # S를 (P,N)로 맞춤
    #                     if axP == 0 and axN == 1:
    #                         S_PN = S
    #                     elif axP == 1 and axN == 0:
    #                         S_PN = S.transpose(0,1)
    #                     else:
    #                         S_PN = S  # fallback

    #                 # 2) "N에 대해서 P 시각화": rows=P, cols=N
    #                 _plot_col_stats(S_PN, name=f"states_B0H0_c{ci:03d}__N_over_P",
    #                                 save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))
    #                 # 1) "P에 대해서 N 시각화": rows=N, cols=P (전치)
    #                 _plot_col_stats(S_PN.transpose(0,1), name=f"states_B0H0_c{ci:03d}__P_over_N",
    #                                 save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))

    #     # =========================================================
    #     # final_states: shape (B, H, P, N)
    #     #   1) "P에 대해 N 시각화" → (rows=N, cols=P)
    #     #   2) "N에 대해 P 시각화" → (rows=P, cols=N)
    #     # =========================================================
    #     if final_states is not None and final_states.ndim == 4:
    #         FS = final_states[0, 0] if (final_states.shape[0]==batch and final_states.shape[1]==nheads) else _fix_B_and_H_or_G(final_states, batch, nheads, ngroups)
    #         # FS 예상: (P, N)
    #         if FS is not None:
    #             # P/N 축 유추 (P=headdim, N=dstate)
    #             if FS.shape != (headdim, dstate):
    #                 if FS.shape == (dstate, headdim):
    #                     FS = FS.transpose(0,1)
    #                 else:
    #                     FS = FS.reshape(headdim, -1) if FS.numel() == headdim*dstate else FS

    #             # 2) "N에 대해 P": rows=P, cols=N
    #             _plot_col_stats(FS, name="final_states_B0H0__N_over_P",
    #                             save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))
    #             # 1) "P에 대해 N": rows=N, cols=P  (전치)
    #             _plot_col_stats(FS.transpose(0,1), name="final_states_B0H0__P_over_N",
    #                             save_dir=PLOT_DIR, colors=("forestgreen","goldenrod","maroon"))

    # except Exception as _e:
    #     print(f"[plot] skipped due to error: {_e}")
    # # ============================================================
    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")


def _quamba2_mamba_chunk_scan_combined_fwd(
        q_x, x_scales, x_head_group_range, x_dim_group_range,
        q_dt, dt_scale, q_A_log, A_log_scale, q_B, B_scale, q_C, C_scale, ssm_state_scale, chunk_size,
        q_D=None, D_scale=None, q_z=None, z_scale=None, dt_bias=None, initial_states=None, seq_idx=None,
        cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), mm_dtype=torch.float16
    ):
    _, _, ngroups, dstate = q_B.shape
    batch, seqlen, nheads, headdim = q_x.shape
    assert len(x_head_group_range.shape) == 2, "x_head_group_range must have shape [n_ssd_group, x_nhead_group]"
    assert len(x_dim_group_range.shape) == 3, "x_dim_group_range must have shape [n_ssd_group, x_nhead_group, n_dim_group]"
    nhead_groups = x_head_group_range.shape[1] # [n_ssd_groups, n_head_groups]
    ndim_groups = x_dim_group_range.shape[2] # [n_ssd_groups, n_head_groups, n_dim_groups]
    assert x_scales.is_cuda
    assert x_head_group_range.is_cuda
    assert x_dim_group_range.is_cuda
    assert x_scales.numel() == ngroups*nhead_groups*ndim_groups, \
            f"{x_scales.numel()} vs. {ngroups}*{nhead_groups}*{ndim_groups}"
    assert x_head_group_range.dtype == torch.int32
    assert x_dim_group_range.dtype == torch.int32

    assert B_scale.is_cuda
    assert B_scale.numel() == ngroups
    assert C_scale.is_cuda
    assert C_scale.numel() == ngroups

    assert nheads % ngroups == 0
    assert q_x.is_cuda
    #assert q_x.dtype == torch.int8
    assert q_x.shape == (batch, seqlen, nheads, headdim)
    assert q_B.is_cuda
    #assert q_B.dtype == torch.int8
    assert q_B.shape == (batch, seqlen, ngroups, dstate)
    assert q_dt.is_cuda
    assert q_dt.shape == (batch, seqlen, nheads)
    assert q_A_log.is_cuda
    assert q_A_log.dtype == torch.int8
    assert q_A_log.shape == (nheads,)
    assert q_C.is_cuda
    #assert q_C.dtype == torch.int8
    assert q_C.shape == q_B.shape
    assert ssm_state_scale.is_cuda
    assert ssm_state_scale.dtype == torch.float32
    assert ssm_state_scale.shape == (ngroups, nhead_groups, ndim_groups, dstate)
    if q_z is not None:
        assert q_z.shape == q_x.shape
    if q_D is not None:
        assert q_D.shape == (nheads, headdim) or q_D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if q_B.stride(-1) != 1:
        q_B = q_B.contiguous()
    if q_C.stride(-1) != 1:
        q_C = q_C.contiguous()
    if q_x.stride(-1) != 1 and q_x.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_x = q_x.contiguous()
    if q_z is not None and q_z.stride(-1) != 1 and q_z.stride(1) != 1:  # Either M or K dimension should be contiguous
        q_z = q_z.contiguous()
    if q_D is not None and q_D.stride(-1) != 1:
        q_D = q_D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    dA_cumsum, dt = _quant_chunk_cumsum_fwd(q_dt, dt_scale, q_A_log, A_log_scale, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _quamba2_chunk_state_fwd(q_B, B_scale, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, mm_dtype=torch.float16, seq_idx=seq_idx, states_in_fp32=True)
    print("OLOLLOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOKOK")
    states, final_states = _quant_state_passing_fwd(
                                rearrange(states, "... p n -> ... (p n)"),
                                dA_cumsum[:, :, :, -1],
                                initial_states=rearrange(initial_states, "... p n -> ... (p n)") \
                                    if initial_states is not None else None,
                                seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=mm_dtype
                            )
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    CB = _quamba2_bmm_chunk_fwd(q_C, C_scale, q_B, B_scale, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)

    # ===== TENSOR DUMP (before _quamba2_chunk_scan_fwd) =====
    import os, torch
    C_deq = (q_C.to(torch.float32) * C_scale.view(1, 1, C_scale.numel(), 1).to(q_C.device))

    _dump_dir = os.getenv("QUAMBA2_DUMP_DIR", "./quamba2_debug")
    os.makedirs(_dump_dir, exist_ok=True)

    # 텐서 그대로 저장 (CPU로 내림)
    torch.save({
        "CB": CB.detach().cpu(),                 # (B, T, nheads, headdim) 예상
        "states": states.detach().cpu(),         # (..., p, dstate)
        "C_deq": C_deq.detach().cpu(),           # (B, T, ngroups, dstate)
        "q_C": q_C.detach().cpu(),               # int8 원본
        "C_scale": C_scale.detach().cpu(),
        "meta": {
            "mm_dtype": str(mm_dtype),
            "chunk_size": int(chunk_size),
            "dt_softplus": bool(dt_softplus),
            "dt_limit": tuple(dt_limit),
        },
    }, os.path.join(_dump_dir, "pre_chunk_scan_tensors.pt"))
    out, out_x = _quamba2_chunk_scan_fwd(
        CB, q_x, x_scales, x_head_group_range, x_dim_group_range, dt, dA_cumsum, q_C, C_scale, states,
        q_D=q_D, D_scale=D_scale, q_z=q_z, z_scale=z_scale,
        seq_idx=seq_idx, mm_dtype=torch.float16
    )
 
    final_states = _quamba2_quant_ssm_states(final_states, x_head_group_range, x_dim_group_range, ssm_state_scale)
     # ===== TENSOR DUMP (after _quamba2_chunk_scan_fwd) =====
    post_payload = {
        "out": out.detach().cpu(),
        "final_states": final_states.detach().cpu(),
        "meta": {
            "mm_dtype": str(mm_dtype),
            "chunk_size": int(chunk_size),
            "dt_softplus": bool(dt_softplus),
            "dt_limit": tuple(dt_limit),
            "final_states_stage": "post-quant",
        },
    }
    if isinstance(out_x, torch.Tensor):
        post_payload["out_x"] = out_x.detach().cpu()

    torch.save(post_payload, os.path.join(_dump_dir, "post_chunk_scan_tensors.pt"))

    if cu_seqlens is None:
        return out, final_states
    else:
        raise NotImplementedError("Only supports `cu_seqlens=None`")
    
    
    
    
@torch.no_grad()
def quantize_B_int8_per_gn(B_fp: torch.Tensor, B_row_scale_cgn: torch.Tensor) -> torch.Tensor:
    """
    B_fp: (B, S, G, N)   FP 텐서
    B_row_scale_cgn: (G, N)  per-(G,N) 스케일 (커널에서 쓰던 것과 동일)
    return: q_B_int8: (B, S, G, N)  int8
    """
    assert B_fp.dim() == 4 and B_row_scale_cgn.dim() == 2
    B, S, G, N = B_fp.shape
    assert B_row_scale_cgn.shape == (G, N)
    assert B_fp.is_cuda and B_row_scale_cgn.is_cuda

    # (1,1,G,N)로 브로드캐스트 후 per-(G,N) 스케일로 나눔
    inv_scale = (1.0 / B_row_scale_cgn).to(dtype=torch.float32, device=B_fp.device).view(1, 1, G, N)
    x = B_fp.to(torch.float32) * inv_scale

    # round half away from zero
    q = torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))
    q.clamp_(-127, 127)
    return q.to(torch.int8)


def quantize_int8_per_group(t: torch.Tensor, scales_g: torch.Tensor, group_dim: int):
    """
    per-G 양자화 전용:
      q = round(t / S_g), clamp to int8
    t: (..., G, ...)
    scales_g: (G,)  혹은 실수로 (G,N) 들어오면 max 축약해서 (G,)로 사용
    return: (q_int8, S_g_float32)  # S_g는 (G,)
    """
    assert t.is_cuda and scales_g.is_cuda
    # (G,N)로 들어오면 per-G로 축약
    if scales_g.dim() == 2:
        scales_g = scales_g.max(dim=-1).values.contiguous()
    assert scales_g.dim() == 1, "이 버전은 per-G 스케일만 지원합니다."
    G = t.shape[group_dim]
    assert scales_g.shape[0] == G, f"스케일 길이({scales_g.shape[0]}) != G({G})"

    view = [1] * t.dim()
    view[group_dim] = G
    S = scales_g.to(dtype=t.dtype, device=t.device).view(*view)  # (..., G, ...)
    q = torch.clamp(torch.round(t / S), -127, 127).to(torch.int8)
    return q, scales_g.to(torch.float32).contiguous()






import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64 }, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64 }, num_stages=5, num_warps=2),
    ],
    key=['B', 'S', 'G', 'N'],
)
@triton.jit
def _quantize_int8_per_g_kernel(
    t_ptr, q_ptr, scales_g_ptr,
    # sizes
    B: tl.constexpr, S: tl.constexpr, G: tl.constexpr, N: tl.constexpr,
    # strides for t: (B,S,G,N)
    st_b, st_s, st_g, st_n,
    # strides for q: (B,S,G,N)
    sq_b, sq_s, sq_g, sq_n,
    # meta
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    """
    Tiles over M=(B*S) rows and N cols for a fixed group g.
    Grid:
      axis0: tiles over (M,N)
      axis1: group g in [0..G)
    """
    # program ids
    pid_g = tl.program_id(axis=1)  # group id
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) %  num_pid_n

    # bounds / offsets
    M = B * S
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask   = mask_m[:, None] & mask_n[None, :]

    # decompose m -> (b, s)
    b_idx = offs_m // S
    s_idx = offs_m - b_idx * S  # offs_m % S

    # base ptrs (broadcast into tile)
    t_ptrs = (t_ptr
              + b_idx[:, None] * st_b
              + s_idx[:, None] * st_s
              + pid_g          * st_g
              + offs_n[None, :] * st_n)
    q_ptrs = (q_ptr
              + b_idx[:, None] * sq_b
              + s_idx[:, None] * sq_s
              + pid_g          * sq_g
              + offs_n[None, :] * sq_n)

    # load per-G scale (float32), guard against zeros
    s_g = tl.load(scales_g_ptr + pid_g).to(tl.float32)
    eps = 1e-8
    s_g = tl.maximum(s_g, eps)

    # hints
    tl.multiple_of(offs_m, 16)
    tl.multiple_of(offs_n, 16)

    # load, scale, quantize (half-away-from-zero), clamp, store
    t = tl.load(t_ptrs, mask=mask, other=0.0).to(tl.float32)  # robust quant: do math in fp32
    x = t / s_g
    qf = tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))
    qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)
    q8 = tl.cast(qf, tl.int8)
    tl.store(q_ptrs, q8, mask=mask)
    
    
@torch.no_grad()
def quantize_int8_per_group_triton(t: torch.Tensor, scales_g: torch.Tensor, group_dim: int):
    """
    Triton-optimized per-G quantization for (B,S,G,N) with group_dim==2.
    q = round(t / S_g), clamp to int8.
    Returns: (q_int8, S_g_float32)
    """
    assert t.is_cuda and scales_g.is_cuda, "t, scales_g must be CUDA tensors"
    assert t.dim() == 4 and group_dim == 2, "This kernel supports shape (B,S,G,N) with group_dim=2"
    B, S, G, N = t.shape

    # Prepare per-G scale (G,)
    if scales_g.dim() == 2:
        # (G,N) -> (G,) by max over N   (your original policy)
        scales_g = scales_g.max(dim=-1).values.contiguous()
    assert scales_g.dim() == 1 and scales_g.shape[0] == G, f"scales_g must be (G,), got {tuple(scales_g.shape)}"

    # Output allocation
    q = torch.empty_like(t, dtype=torch.int8)

    # Launch
    grid = lambda META: (
        triton.cdiv(B * S, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
        G
    )

    # Make sure we pass strides in element units (PyTorch gives strides in elements already)
    _quantize_int8_per_g_kernel[grid](
        t, q, scales_g.to(dtype=torch.float32),
        B, S, G, N,
        t.stride(0), t.stride(1), t.stride(2), t.stride(3),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
    )

    return q, scales_g.to(torch.float32).contiguous()




##### torch quantization for compensation calibration #####
import torch

@torch.no_grad()
def _round_half_away_from_zero(x: torch.Tensor):
    # Triton 커널과 동일한 half-away-from-zero
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))

@torch.no_grad()
def quantize_int8_per_hp_torch(x: torch.Tensor, s_hp: torch.Tensor):
    """
    x: (B,S,H,P) CUDA
    s_hp: (H,P) CUDA  (float32 권장)
    return: q_x(int8, B,S,H,P), s_hp(float32, contiguous)
    """
    assert x.is_cuda and x.dim() == 4
    H, P = x.shape[2], x.shape[3]
    assert s_hp.is_cuda and tuple(s_hp.shape) == (H, P)

    s_hp = s_hp.to(torch.float32).clamp_min_(1e-8).contiguous()
    scale = s_hp.view(1, 1, H, P)                      # (1,1,H,P)
    qf = _round_half_away_from_zero(x.to(torch.float32) / scale)
    q  = torch.clamp(qf, -127, 127).to(torch.int8)
    return q, s_hp


@torch.no_grad()
def quantize_int8_per_group_torch(
    t: torch.Tensor,
    scales_g: torch.Tensor,
    group_dim: int = 2,
    eps: float = 1e-8,
):
    """
    Pure-PyTorch per-group int8 quantization for tensors shaped (B, S, G, N) with group_dim == 2.

    Args:
        t:          float tensor (B, S, G, N). CUDA or CPU 모두 가능.
        scales_g:   (G,) 또는 (G, N). (G, N)인 경우 dim=-1 max로 (G,)로 축약.
        group_dim:  반드시 2 (G 차원).
        eps:        0 스케일 가드.

    Returns:
        q_int8:     torch.int8 tensor, same shape/device as t.
        s_g:        torch.float32 tensor of shape (G,), contiguous.
    """
    assert t.dim() == 4 and group_dim == 2, "This function supports shape (B,S,G,N) with group_dim=2"
    B, S, G, N = t.shape

    # 스케일 형태 정규화
    if scales_g.dim() == 2:
        # (G, N) -> (G,) by max over N  (원본 정책과 동일)
        scales_g = scales_g.max(dim=-1).values
    assert scales_g.dim() == 1 and scales_g.numel() == G, f"scales_g must be (G,), got {tuple(scales_g.shape)}"

    # dtype/디바이스 정렬
    device = t.device
    s_g = scales_g.to(device=device, dtype=torch.float32).contiguous()
    s_g = torch.clamp(s_g, min=eps)  # 0 가드

    # 브로드캐스팅 모양: (1,1,G,1)
    s_g_bc = s_g.view(1, 1, G, 1)

    # 연산은 안정성을 위해 fp32로
    x = (t.to(torch.float32) / s_g_bc)

    # half-away-from-zero 반올림: x>=0 ? floor(x+0.5) : ceil(x-0.5)
    qf_pos = torch.floor(x + 0.5)
    qf_neg = torch.ceil(x - 0.5)
    qf = torch.where(x >= 0, qf_pos, qf_neg)

    # int8 클램프 [-127, 127]
    qf = torch.clamp(qf, -127.0, 127.0)

    # int8 캐스트
    q = qf.to(torch.int8)

    return q, s_g