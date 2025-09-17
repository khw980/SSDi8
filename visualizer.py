
import torch
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect)
from pathlib import Path
from einops import rearrange
from typing import Dict, Any, Literal
from functools import wraps
_cached: Dict[str, Any] = {}          # forward hook 가 채움

def register_basic_hooks(model, *, batch_idx=0,
                         head_idx=None, gp_idx=None, verbose=True):
    mx = getattr(model.backbone.layers[-1], "mixer",
                 model.backbone.layers[-1])
    
    _cached.update(
        nheads  = mx.nheads,
        headdim = mx.headdim,
        ngroups = mx.ngroups,
        dstate  = mx.d_state,
        batch_idx = batch_idx,
        captured_in = False,   # ➊ in-proj 완료 여부
        frozen      = False,   # ➋ 모든 tensor 저장 완료 여부
    )
    nH, P  = mx.nheads, mx.headdim
    G, N   = mx.ngroups, mx.d_state
    d_ssm  = nH * P
    d_BC   = 2 * G * N
    d_proj = 2*d_ssm + d_BC + nH          # z x B C dt

    # ───────────── in-proj (z/x/B/C/dt) ─────────────
    def hook_in(_m, _inp, proj_out):
        if _cached["captured_in"]:        # 이미 한 번 받은 뒤면 skip
            return

        zxbcdt = proj_out.detach().cpu()[batch_idx]
        z, x, BC, dt = torch.split(zxbcdt,
                                   [d_ssm, d_ssm, d_BC, nH], dim=-1)

        z = rearrange(z, "l (h p)->l h p", h=nH)
        x = rearrange(x, "l (h p)->l h p", h=nH)
        B, C = torch.split(BC, [G*N, G*N], dim=-1)
        B = rearrange(B, "l (g n)->l g n", g=G)
        C = rearrange(C, "l (g n)->l g n", g=G)
        dt = dt.abs()                     # (L, nH)

        if head_idx is not None:
            z, x = z[:, head_idx], x[:, head_idx]
            dt   = dt[:, head_idx].unsqueeze(1)  # (L,1)
        if gp_idx is not None:
            B, C = B[:, gp_idx], C[:, gp_idx]

        _cached.update(z=z.abs(), x=x.abs(),
                       b=B.abs(), c=C.abs(), dt=dt,
                       captured_in=True)        # ➌ 플래그 ON

        if verbose:
            print(f"[in-proj] z{z.shape} x{x.shape} "
                  f"B{B.shape} C{C.shape} dt{dt.shape}")

        h_in.remove()                            # in-proj hook 해제

    h_in = mx.in_proj.register_forward_hook(hook_in)

    # ───────────── mixer 내부 Y (ssd_out) ─────────────
    def hook_y(_m, _inp, y_ssd):
        if not _cached["captured_in"] or _cached["frozen"]:
            return
        y_ssd = y_ssd.detach().cpu()[batch_idx]
        y_ssd = rearrange(y_ssd, "l (h p)->l h p", h=nH)
        if head_idx is not None:
            y_ssd = y_ssd[:, head_idx]
        _cached["y"] = y_ssd.abs()
        if verbose:
            print(f"[ssd_out] y{y_ssd.shape}")
        h_y.remove()          # hook 자체 제거

    h_y = (getattr(mx, "ssd_out_act", None) or mx
           ).register_forward_hook(hook_y)



    def hook_out2(_m, _inp, out_tensor, *, topk: int = 100):
        """
        - out_tensor : (B, L, D)   ─ Mixer 출력
        - topk       : 가장 큰 mean 값 N 개만 보고 싶을 때
        """
        if _cached.get("done"):          # 이미 한 번 찍었으면 PASS
            return

        B = out_tensor.size(0)
        if B < 3:                        # 안전 가드
            print(f"[hook_out2] batch={B} → 3개 미만, skip")
            return

        # 두 번째·세 번째 배치의 (L,D) → dim-별 평균 (D,)
        mean1 = out_tensor.detach().cpu()[1].mean(dim=0)
        mean2 = out_tensor.detach().cpu()[2].mean(dim=0)

        # ── 상위 top-k 값과 인덱스만 추출 ─────────────────────
        vals1, idxs1 = torch.topk(mean1, k=topk)   # (topk,)
        vals2, idxs2 = torch.topk(mean2, k=topk)

        print(f"\n[mixer-out batch-1] top-{topk}")
        for i, v in zip(idxs1.tolist(), vals1.tolist()):
            print(f"dim {i:>5d} : {v:.6f}")

        print(f"\n[mixer-out batch-2] top-{topk}")
        for i, v in zip(idxs2.tolist(), vals2.tolist()):
            print(f"dim {i:>5d} : {v:.6f}")

        _cached["done"] = True
        h_out2.remove()                  # 한 번만 실행되도록 hook 제거
    h_out2 = mx.register_forward_hook(lambda m,i,o: hook_out2(m,i,o, topk=100))

    def hook_out(_m, _inp, out_tensor):
        if not _cached["captured_in"] or _cached["frozen"]:
            return
        out = out_tensor.detach().cpu()[batch_idx]
        mean_per_d = out.mean(dim=(0, 1))    # shape: (D,)
        print(mean_per_d)                    # tensor([...], dtype=torch.float32)
        out = rearrange(out, "l (h p)->l h p", h=nH)
        if head_idx is not None:
            out = out[:, head_idx]
        _cached["out"] = out.abs()
        if verbose:
            print(f"[mixer out] out{out.shape}")

        _cached["frozen"] = True   # ➍ 모든 tensor 확보 완료
        h_out.remove()


    h_out = mx.register_forward_hook(hook_out)
from matplotlib import pyplot as plt, colors
import torch

def _bar3d(
    ax,
    mat: torch.Tensor,
    *,
    cmap="coolwarm",
    alpha=0.9,
    scale=1.0,
    width=0.5,
    zmax=20.0
):
    """3-D bar plot ― 얇은 ‘실선’ 스타일."""
    mat  = mat.abs() * scale
    L, D = mat.shape

    xs, ys = torch.meshgrid(torch.arange(L), torch.arange(D), indexing="ij")
    vals   = mat.flatten().numpy()                # torch → numpy

    cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    norm    = colors.Normalize(vmin=vals.min(), vmax=max(vals.max(), 1e-8))
    facecol = cmap_fn(norm(vals))                 # RGBA 색상

    ax.bar3d(
        xs.flatten() + 0.5 - width/2,
        ys.flatten() + 0.5 - width/2,
        0.0,
        width, width, vals,                      # dx, dy, dz
        color=facecol,
        shade=False,
        edgecolor="none",
        linewidth=0.0,
        alpha=alpha,
    )

    # 축 레이블   최소화
    ax.set_xlabel("Token", labelpad=4)
    ax.set_ylabel("Dims",  labelpad=4)
    ax.set_zlabel("Value", labelpad=4)
    ax.grid(False)


    

def plot_tensor(name: Literal["X","Z","Y","B","C","dt","OUT"],
                *, mode: Literal["3d","heat"]="3d",
                flatten: bool=False,   # ★ 새 옵션
                save_dir: str | Path = "head_vis",
                cmap="viridis", alpha=.85, scale=1.0):

    key = name.lower()
    if key not in _cached:
        raise RuntimeError(f"{name} 가 _cached 에 없습니다.")

    t = _cached[key]          # ── 원본 Tensor
    if t.ndim == 2:           # (L,D) → (L,1,D)  ← dt, head slice 등
        t = t.unsqueeze(1)

    # ────────────── ① Head·P 축을 한꺼번에 볼 때 ──────────────
    if flatten and t.ndim == 3:               # (L, H, P) or (L,G,N)
        t = t.reshape(t.shape[0], 1, -1)      # → (L, 1, H*P)
                                             #   ↳ 더 이상 loop 필요 X
    L, M, D = t.shape
    label = "Head×P" if flatten else (
            "Head" if name.lower() in {"x","z","y","dt","out"} else "Group"
    )

    base = Path(save_dir, name.upper())
    base.mkdir(parents=True, exist_ok=True)

    # ────────────── ②  그리기 ──────────────
    for m in range(M):
        mat = t[:, m]        # (L,D)   D = H*P  (flatten)  or  P/N

        if mode == "heat":
            fig, ax = plt.subplots(figsize=(6,3))
            im = ax.imshow(mat.T, aspect="auto", origin="lower",
                           interpolation="nearest", cmap=cmap)
            ax.set(xlabel="token (L)", ylabel="dim",
                   title=f"{name.upper()} – {label} {m}")
            fig.colorbar(im, ax=ax, fraction=.025)

        else:                 # 3-D bar
            fig = plt.figure(figsize=(8,4))
            ax  = fig.add_subplot(111, projection="3d")
            _bar3d(ax, mat, cmap=cmap, alpha=alpha, scale=scale)
            ax.set(xlabel="token", ylabel="dim", zlabel="value",
                   title=f"{name.upper()} – {label} {m}")

        fname = base / f"{name.lower()}_{m}.png"
        fig.tight_layout(); fig.savefig(fname, dpi=160); plt.close(fig)
        print("✓ saved →", fname)


# ───────────────────────────────────────────────
#   통계 플롯:  median-99%-max   (dim ↦ value)
# ───────────────────────────────────────────────
import numpy as np

import numpy as np          # ← xs 생성을 위해 필요
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Literal

# def plot_dim_stats(
#     name: Literal["X","Z","Y","OUT","B","C","dt"],
#     *,
#     save_dir: str | Path = "head_vis",
#     colors = ("forestgreen", "goldenrod", "maroon"),   # 50 % · 99 % · max
#     dpi    = 160,
# ):
#     key = name.lower()
#     if key not in _cached:
#         raise RuntimeError(f"{name} 가 _cached 에 없습니다 – forward 먼저!")

#     t = _cached[key]                    # ex) (L,H,P) | (L,G,N) | (L,H)
#     if t.ndim == 2:
#         t = t.unsqueeze(1)              # (L,1,D) 로 맞춤

#     L, M, D = t.shape
#     mat = t.reshape(L, -1).float()      # ★ int → float32 변환 ★   (L, K)

#     # ── 토큰축(L) 50 · 99 · max 통계 ──────────────────────────
#     median = torch.median(mat, dim=0).values          # 50 %
#     p99    = torch.quantile(mat, 0.99, dim=0)         # 99 %
#     vmax   = torch.max(mat, dim=0).values             # max
#     xs     = np.arange(mat.shape[1])                  # dim index (0 … K−1)

#     # ── 플롯 ────────────────────────────────────────────────
#     fig, ax = plt.subplots(figsize=(7,3), dpi=dpi)
#     ax.fill_between(xs, 0,       median, color=colors[0], alpha=.85, label="median (50 %)")
#     ax.fill_between(xs, median,  p99,    color=colors[1], alpha=.85, label="99 percentile")
#     ax.fill_between(xs, p99, vmax,                     color=colors[2], alpha=.85,    label="max")

#     ax.set(xlabel="Dim index (flattened)", ylabel="Value",
#            title=f"{name} ‒ per-dimension stats")
#     ax.legend(fontsize=8, framealpha=.9)
#     ax.margins(x=0)                      # 좌우 여백 제거

#     out_dir = Path(save_dir, f"{name.upper()}_stats")
#     out_dir.mkdir(parents=True, exist_ok=True)
#     fname = out_dir / f"{name.lower()}_stat.png"
#     fig.tight_layout()
#     fig.savefig(fname)
#     plt.close(fig)
#     print("✓ saved →", fname)


def plot_dim_stats(
    name: Literal["X","Z","Y","OUT","B","C","dt"],
    *,
    save_dir: str | Path = "head_vis",
    colors = ("forestgreen", "goldenrod", "maroon"),  # 50 % · 99 % · max
    dpi    = 160,
):
    key = name.lower()
    if key not in _cached:
        raise RuntimeError(f"{name} 가 _cached 에 없습니다 – forward 먼저!")

    t = _cached[key]                    # (L,H,P) | (L,G,N) | (L,H)
    if t.ndim == 2:
        t = t.unsqueeze(1)              # (L,1,D)

    L, M, D = t.shape
    mat = t.reshape(L, -1).float()      # (L, K), where K = M*D

    # ── dim 축에 대해 통계 계산 (각 토큰별로) ─────────────
    median = torch.median(mat, dim=1).values          # (L,)
    p99    = torch.quantile(mat, 0.99, dim=1)         # (L,)
    vmax   = torch.max(mat, dim=1).values             # (L,)
    xs     = np.arange(L)                             # token index

    # ── 플롯 ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7,3), dpi=dpi)
    ax.fill_between(xs, 0,       median, color=colors[0], alpha=.85, label="median (50 %)")
    ax.fill_between(xs, median,  p99,    color=colors[1], alpha=.85, label="99 percentile")
    ax.fill_between(xs, p99,     vmax,   color=colors[2], alpha=.85, label="max")

    ax.set(xlabel="Token index (L)", ylabel="Value",
           title=f"{name} ‒ per-token stats")
    ax.legend(fontsize=8, framealpha=.9)
    ax.margins(x=0)

    out_dir = Path(save_dir, f"{name.upper()}_per_token")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{name.lower()}_per_token_stat.png"
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
    print("✓ saved →", fname)
