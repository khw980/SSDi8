from lowrank import LowRankBranch
import torch, torch.nn as nn
class Had2OutBranchHook_forsmooth(nn.Module):
    """
    had_module → 입력(fp16) 캐시
    out_module → branch 결과(fp16) 더하기
    """
    def __init__(self, had_module: nn.Module, out_module: nn.Module, branch: LowRankBranch):
        super().__init__()
        self.branch   = branch
        self._cache_x = None

        # 1️⃣ had 입력 캐시 (pre‑hook)
        def _cache_input(mod, args, kwargs):
            self._cache_x = args[0]      # fp16 입력 텐서

        self.had_handle = had_module.register_forward_pre_hook(
            _cache_input, with_kwargs=True
        )

        # 2️⃣ out 출력에 누적 (post‑hook)
        def _add_branch(mod, args, kwargs, output):
            if self._cache_x is None:
                return output            # 안전장치

            y_low        = self.branch(self._cache_x)  # fp16 → fp16
            self._cache_x = None                        # 캐시 비우기

            # out_proj 도 fp16 → in‑place add
            if isinstance(output, torch.Tensor):
                return output.add(y_low)

            # tuple / list / dict 형태를 쓰지 않는다면 아래는 사실상 불필요
            return output

        self.out_handle = out_module.register_forward_hook(
            _add_branch, with_kwargs=True
        )

    def remove(self):
        self.had_handle.remove()
        self.out_handle.remove()






class Had2OutBranchHook(nn.Module):
    """
    had_module     : (ex) qmixer.had          – 입력 텐서를 캐싱
    had_low_module : (ex) qmixer.had_low      – 입력을 한 번 더 변환
    out_module     : (ex) qmixer.out_proj     – 출력에 누적(add)
    branch         : LowRankBranch(LoRA)      – had_low 결과를 저차 변환
    """
    def __init__(
        self,
        had_module: nn.Module,
        had_low_module: nn.Module,
        out_module: nn.Module,
        branch: LowRankBranch,
    ):
        super().__init__()
        self.branch        = branch
        self.had_low       = had_low_module   # fp16 연산 가정
        self._cache_x_fp16 = None             # 입력 캐시

        # ① had 입력을 캐싱 (pre‑hook)
        def _cache_input(mod, args, kwargs):
            self._cache_x_fp16 = args[0]      # 이미 fp16

        self.had_handle = had_module.register_forward_pre_hook(
            _cache_input, with_kwargs=True
        )

        # ② out_proj 출력에 저차 결과 누적 (post‑hook)
        def _add_branch(mod, args, kwargs, output):
            if self._cache_x_fp16 is None:
                return output                 # 안전장치

            # had_low 변환 → Low‑rank 변환
            x_low = self.had_low(self._cache_x_fp16)   # fp16 → fp16
            
            y_low = self.branch(x_low)                 # (B, L, D_out)
            self._cache_x_fp16 = None                  # 캐시 비우기

            # out_proj 도 fp16 이므로 in‑place add
            # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
            # print(output.norm(), "XR norm")
            # print(output, "XR norm")
            # torch.set_printoptions(profile='default')
            # print(y_low.norm(), "XL norm")
            # print(y_low, "XL")
            output=output.add(y_low)
            # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
            # print(output.norm(), "XW norm")
            # print(output, "XW")
            # torch.set_printoptions(profile='default')
            return output

        self.out_handle = out_module.register_forward_hook(
            _add_branch, with_kwargs=True
        )

    # 필요 시 훅 해제용
    def remove(self):
        self.had_handle.remove()
        self.out_handle.remove()