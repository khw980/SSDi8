import torch
import torch.nn as nn
from functools import partial
import logging
from datasets import load_dataset
from tqdm import tqdm
from quamba.qLinearLayer import W8A8B16O16Linear

# import torch, matplotlib.pyplot as plt, numpy as np
# from pathlib import Path
# import datetime, re, os

# # ─────────────────────────────────────────────
# # ① 헬퍼: 파일명 슬러그
# # ─────────────────────────────────────────────
# def _slugify(text: str) -> str:
#     return re.sub(r"[^0-9a-zA-Z_-]+", "_", text.strip()).strip("_")

# # ─────────────────────────────────────────────
# # ② 저장 루트 디렉터리 (현재 작업 dir 하위)
# # ─────────────────────────────────────────────
# PLOT_DIR = Path.cwd() / "smoothing_weight_plot"
# PLOT_DIR.mkdir(exist_ok=True)

# # ─────────────────────────────────────────────
# # ③ 메인 함수
# # ─────────────────────────────────────────────
# def plot_weight_surface_full(
#     weight: torch.Tensor,
#     title: str = "weight_full",
#     *,
#     mode: str = "auto",      # "auto" | "3d" | "heat"
#     dpi: int = 160,
#     cmap: str = "inferno",
#     alpha: float = 0.85,
#     scale: float = 1.0,
#     bar_width: float = .5,    # 3-D 막대 굵기
#     threshold: int = 512*512, # auto 전환 기준(행*열)
# ):
#     """
#     Parameters
#     ----------
#     weight : torch.Tensor  (out_dim, in_dim)
#     title  : 그래프·파일 제목
#     mode   : "3d" → bar3d / "heat" → 2-D / "auto" → threshold 기준 자동
#     dpi    : 저장 해상도
#     cmap   : 컬러맵
#     alpha  : 투명도 (3-D 막대)
#     scale  : 값 스케일 (3-D 막대 높이 조정)
#     """
#     # ── 파일명 --------------------------------------------------
#     base = _slugify(title.lower()) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     fname = PLOT_DIR / f"{base}.png"
#     idx = 1
#     while fname.exists():              # 중복 시 _1, _2 …
#         fname = PLOT_DIR / f"{base}_{idx}.png"
#         idx += 1

#     # ── 텐서 준비 ---------------------------------------------
#     W = weight.detach().cpu().abs() * scale     # (H,W)
#     H, Wd = W.shape

#     # ── 모드 결정 ---------------------------------------------
#     if mode == "auto":
#         mode = "heat" if (H * Wd > threshold) else "3d"

#     # ── 그리기 -------------------------------------------------
#     if mode == "heat":
#         fig, ax = plt.subplots(figsize=(7, 4.5), dpi=dpi)
#         im = ax.imshow(W.numpy(), aspect="auto", origin="lower", cmap=cmap)
#         ax.set(xlabel="in_dim", ylabel="out_dim", title=title)
#         fig.colorbar(im, ax=ax, fraction=.025)
#     else:  # "3d"
#         from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
#         from matplotlib import colors
#         fig = plt.figure(figsize=(8, 4), dpi=dpi)
#         ax  = fig.add_subplot(111, projection="3d")

#         xs, ys = torch.meshgrid(torch.arange(H), torch.arange(Wd), indexing="ij")
#         vals   = W.flatten().numpy()

#         cmap_fn = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
#         norm    = colors.Normalize(vmin=vals.min(), vmax=max(vals.max(), 1e-8))
#         facecol = cmap_fn(norm(vals))

#         ax.bar3d(
#             xs.flatten() + 0.5 - bar_width/2,
#             ys.flatten() + 0.5 - bar_width/2,
#             0.0,
#             bar_width, bar_width, vals,
#             color=facecol, shade=False,
#             edgecolor="none", linewidth=0.0, alpha=alpha,
#         )
#         ax.set(xlabel="Token / out_dim", ylabel="in_dim", zlabel="|W|", title=title)
#         ax.grid(False)

#     fig.tight_layout()
#     fig.savefig(fname, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ saved → {fname}")
    
    

class SmoothModule(nn.Module):
    def __init__(self, weight_to_smooth, tensor_name=None):
        super(SmoothModule, self).__init__()
        self.tensor_name = tensor_name
        self.weight_to_smooth=weight_to_smooth
        self.register_buffer("scales", None)
        self.activated = False
    @torch.no_grad()
    def forward(self, x, reverse=False): # 활성화에 대한 평활화 계수 곱셈 시행
        #print("weight_to_smooth:", self.weight_to_smooth)
        # print(self.scales)
        assert not torch.isnan(x).any(), "Input tensor x contains NaNs."
        if not self.activated:
            return x
        else:
            if reverse:
                return x.mul(self.scales)                                               ##이거 diagonal인가? 확인
            else:
                return x.div(self.scales)
            
        
    def configure(self, scales): # smooth_mamba에서 호출되어 설정되어 forward에서 사용됨
        self.scales = scales
        assert not torch.isnan(self.scales).any(), "Scales contains NaNs."
        self.activated = True
        

def get_act_scalers_mamba(model, tokenizer,
                num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    
    act_scales = {}
    
    def stat_act_hook(m, inputs, outputs, name):
        x = inputs[0].clone().detach() if isinstance(inputs, tuple) else inputs.clone().detach()
        assert x.dim() == 3, "Assuming x is of input shape (B, L, D)"
        comming_max = x.abs().amax(dim=(0, 1))
        
        if name not in act_scales:
            act_scales[name] = comming_max
        else:
            act_scales[name] = torch.max(act_scales[name], comming_max)
    
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            hooks.append(
                m.register_forward_hook(partial(stat_act_hook, name=name))
            )
            
    logging.info("Prepare for smoothing..")
    calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
    #calibration_dataset = load_dataset("wikitext", data_filed="wikitext-103-v1", split="train")
    calibration_dataset.shuffle(seed=42)
    logging.info("Run smoothing calibration")
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(calibration_dataset[i]["text"], return_tensors="pt",
                              max_length=seq_len, truncation=True).input_ids.to(device)
        model(input_ids)
    
    for h in hooks:
        h.remove()


    return act_scales


@torch.no_grad()
def smooth_fc(weight, act_scale, alpha=0.5): # 가중치에 대해 평활화 계수 곱셈 시행
    # print("\n")
    # print(weight)
    # print("start")
    # orig_W = weight.detach().clone()
    # plot_weight_surface_full(orig_W, "W plot", mode="3d")
    
    device = weight.device
    dtype = weight.dtype
    act_scale = act_scale.to(device).to(dtype)
    # linear fc weight shape [out_dim, in_dim]
    weight_scale = weight.abs().max(dim=0, keepdim=True)[0].clamp(min=1e-5) # [out_dim, in_dim] -> [1, in_dim]
    # print("immediate")
    if act_scale.dim() == 0:
        sm_scale = (act_scale[None].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)
    else:
        sm_scale = (act_scale[None, :].pow(alpha) / weight_scale.pow(1-alpha)).clamp(
            min=1e-5).to(device).to(dtype)
    weight = weight.mul_(sm_scale)  # Apply the smooth scale to the weight
    # print(weight)
    # smoothing_W = weight.detach().clone()
    # plot_weight_surface_full(smoothing_W, "Smoothing w plot", mode="3d")
    # print("end")
    return sm_scale

def smooth_mamba(model, tokenizer, num_samples=512, seq_len=512, alpha=0.7, 
                 inp_smoothing = False, out_smoothing=False, inp_alpha=0.8, out_alpha=0.8):
    
    act_scales = get_act_scalers_mamba(model, tokenizer, num_samples, seq_len)
    #TODO: Calculate the real act scales with linear layers.
    smooth_scales = {}
    for name, m in model.named_modules():
        if isinstance(m, SmoothModule): # 모든 서브모듈을 돌며 SmoothModule에 대해 평활화 적용
            if(((m.weight_to_smooth == "out_proj" and inp_smoothing) or (m.weight_to_smooth == "in_proj" and out_smoothing)) and inp_smoothing != out_smoothing) : continue

            name_prefix = ".".join(name.split(".")[:-1])
            weight_name = name_prefix + "." + m.weight_to_smooth

            # smoothmodule contains weight name for the
            # corresponding QLinearLayer
            ############ 수정함 ###########
            weight_module = model.get_submodule(weight_name) # SmoothModule이 연결되는 가중치 텐서를 pick
            ############ 수정함 ###########
            original_weight = weight_module.weight
            scale = act_scales[name]
            if m.weight_to_smooth == "out_proj":
                sm_scale = smooth_fc(original_weight, scale, alpha=out_alpha)
            elif m.weight_to_smooth == "in_proj":
                sm_scale = smooth_fc(original_weight, scale, alpha=inp_alpha)
            smooth_scales[name] = sm_scale
            # m.weight = smooth_weight

            # logging.info(f"Configure smooth module {name}")
            # m.configure(smooth_scales[name])

    for name, m in model.named_modules():
        if isinstance(m, SmoothModule):
            logging.info(f"Configure smooth module {name}")
            m.configure(smooth_scales[name]) # 여기서 활성화에 대한 smooth_scale get