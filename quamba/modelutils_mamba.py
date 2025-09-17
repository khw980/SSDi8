import os
import gc
import copy
import logging
from tqdm import tqdm
from functools import partial
import json

import torch
import torch.nn as nn
from datasets import load_dataset

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.utils.generation import InferenceParams
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

from quamba.quamba_mixer_seq import QuambaLMHeadModel
from .qEmbedding import W4O16Embedding, W8O16Embedding
from .qLinearLayer import HadLinear, W4A16B16O16Linear, W4A8B16O16Linear, W8A8B16O16Linear
from .qActLayer import ActIdentity
from .qMamba2 import Mamba2Simple, W4A8QMamba2, W4A16QMamba2, W8A8QMamba2
from .qMambaLayer import MambaSimple, W4A8QMamba, W4A16QMamba, W8A8QMamba
from .qHadamard import Hadamard
from .qBlock import HybridQuambaBlock
# from .fusedNorm import FusedRMSNorm
from .qNorm import QRMSNorm
from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver
from .observer import PerSSDGroupObserver, CrossHeadMinmaxObserver, _ChannelMeanCollector
from .observer import CachedStatesCrossHeadMinmaxObserver
from .observer import ChunkCollector
from .gptq_utils import GPTQ
from .reorder_utils import get_reorder_params, reorder_mamba
from .hadamard_utils import had_transform
from .data_loaders import get_loaders
from .smooth_quant_utils import smooth_mamba
from lowrank import LowRankBranch  


logger = logging.getLogger(__name__)

@torch.no_grad()
def fuse_ln_linear(norm, linear) -> None:
    """
    fuse the layernorm weight to the adjacent linear layer.
    """
    linear_dtype = linear.weight.dtype

    # Calculating new weight and bias
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)  
    if hasattr(norm, 'bias') and norm.bias is not None:
        if linear.bias is None:
            linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
        linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
        linear.bias.data = linear.bias.data.to(linear_dtype)
    # Reset the learnable weight in RMSNorm to 1
    norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1
import time
from contextlib import contextmanager
@contextmanager
def _time_block(label: str, device=None, sync=True):
            """CUDA면 cudaEvent로, 아니면 perf_counter로 측정"""
            use_cuda = device is not None and torch.cuda.is_available()
            if use_cuda:
                if sync:
                    torch.cuda.synchronize(device)
                start = torch.cuda.Event(enable_timing=True)
                end   = torch.cuda.Event(enable_timing=True)
                start.record()
                try:
                    yield
                finally:
                    end.record()
                    if sync:
                        torch.cuda.synchronize(device)
                    ms = start.elapsed_time(end)
                    print(f"[TIME] {label:<24} {ms:.3f} ms")
            else:
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    t1 = time.perf_counter()
                    print(f"[TIME] {label:<24} {(t1 - t0)*1000:.3f} ms")
@torch.no_grad()
def configure_model(model, model_type, use_had_transform=True, inp_smoothing=False,out_smoothing=False):
    device = next(model.parameters()).device
    if model_type == "mamba":
        # process embedding and lm_head
        if use_had_transform:            
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) 
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            torch.cuda.empty_cache()
        layers = model.backbone.layers
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                m = MambaSimple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        # process embedding and lm_head
        if use_had_transform:            
            # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
            lm_head_clone = model.lm_head.weight.data.clone()
            # transform embedding first
            model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) #입력 임베딩 회전
            # do layernorm fusion to lm_head and then transform
            model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
            ##LM-Head 와 RMSNorm 합치기
            model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
            ##RMSNorm 합쳤으니까 1로만들기
            model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
            ##다시한번 회전
            torch.cuda.empty_cache()
        # process layers
        
        layers = model.backbone.layers
        #######################################
        # print("layers=",layers)              
        # print("lenlayers=",len(layers))   64, 2.7B -> layers=Blocks
        # print("mdoel=,",model)
        # for i, blk in enumerate(model.backbone.layers):        # 64개 Block 순회
        #     n_param = sum(p.numel() for p in blk.parameters())
        #     print(f"{i:02d} │ {blk.__class__.__name__:<10} │ {n_param/1e6:7.2f} M params")
        # for name, module in model.named_modules():
        #     print(f"{name:40s}  →  {module.__class__.__name__}")

        ######################################
        for i in range(len(layers)):
            if isinstance(layers[i], Block):
                # fuse norm to in_proj first
                fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)  ###RMSNorm 을 inprj에 합치기
                # use simplied mamba block to get the scaling factors
                # from linear layers without pain
                ##########################################################LOWRANK 적용 유력 시점: RMS만 합쳐진 상태 ########################################
                m = Mamba2Simple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform, inp_smoothing=inp_smoothing,out_smoothing=out_smoothing).to(device)
                layers[i].mixer = m
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    model.config.use_cache = False
    with _time_block("eval_latency!!!!!!!", device=device):
        model.eval()

    print("mdoel=,",model)
    return model
######################    Mabma2  래핑 ############################
@torch.no_grad()
def run_quamba_calibration(
        model, model_type, tokenizer, num_samples=512, seq_len=2048,
        calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_proj"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        is_x = lambda op: op == "x_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    fp_proj_mean_hooks = {}
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        mixer = get_mamba(layers[i])
        x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        ori_x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        B_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        C_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        state_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        cb_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
        
        ##DO
        _ssd_col = _ChannelMeanCollector()
        _out_col = _ChannelMeanCollector()

        h_ssd = mixer.ssd_out_act.register_forward_hook(
            lambda _m, _i, out, col=_ssd_col: col.update(out)
        )
        h_out = mixer.out_proj.register_forward_hook(
            lambda _m, _i, out, col=_out_col: col.update(out)
        )
        fp_proj_mean_hooks[f"L{i}/ssd_out"] = (_ssd_col, h_ssd)
        fp_proj_mean_hooks[f"L{i}/out_proj_out"] = (_out_col, h_out)
        
        ### COMP ###


        # mixer에 달아두기 → forward에서 collect_obs로 전달됨
        mixer._collect_obs = {"ori_x_chp": ori_x_col,"x_chp": x_col, "B_cgn": B_col,"C_cali":C_cali,"state_cali":state_cali,"cb_cali":cb_cali}
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op) or is_ssm_state(op):
                    observers[i][op + ":input"] = PerTensorPercentileObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        percentile_alpha=percentile_alpha
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )

    device = next(model.parameters()).device
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        del inf_cache
    
    for h in hooks:
        h.remove()
        
    
    fp_means = {}
    for k,(collector,handle) in fp_proj_mean_hooks.items():
        handle.remove()
        fp_means[k] = collector.mean
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale.to(torch.float32)
        if i < len(layers):
            col = getattr(get_mamba(layers[i]), "_collect_obs", None)
            if col is not None:
                act_scales[i]["chunk_state:x_scale_chp"] = col["x_chp"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_state:ori_x_scale_chp"] = col["ori_x_chp"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_state:B_scale_cgn"] = col["B_cgn"].get_scale()  # (C,G,N)
                act_scales[i]["chunk_scan:C_scale"] = col["C_cali"].get_scale()  # (C,H,P)
                act_scales[i]["chunk_scan:cb_scale"] = col["cb_cali"].get_scale()  # (C,H,P)
                act_scales[i]["ssd_combined:state_scale"] = col["state_cali"].get_scale()  # (C,H,P)
                # 메모리/부작용 방지
                delattr(get_mamba(layers[i]), "_collect_obs")
    del observers
    return act_scales, fp_means

@torch.no_grad()
def run_quamba2_calibration(
        model, model_type, tokenizer, reorder_params,
        num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None
    ):

    if model_type == "mamba":
        raise NotImplementedError("Not support for mamba")
    elif model_type == "mamba2":
        layers = model.backbone.layers
        is_traget_block = lambda block: isinstance(block, Block)
        get_mamba = lambda block: block.mixer
        is_x = lambda op: op == "x_conv_out"
        is_BC = lambda op: op == "B_conv_out" or op == "C_conv_out"
        is_ssm_state = lambda op: op == "ssm_state_act"
        is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
        percentile_alpha=0.99999
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2'")

    # register min/max observers, num_layer + lm_head
    observers = [{} for _ in range(len(layers) + 1)]
    
    def stat_hook(m, inputs, outputs, op, block_idx):
        # register the new information to observer
        if isinstance(inputs, tuple):
            inputs = inputs[0]
        observers[block_idx][op + ":input"].update(inputs.clone().detach())

        if isinstance(outputs, tuple):
            outputs = outputs[0]
        observers[block_idx][op + ":output"].update(outputs.clone().detach())

    hooks = []
    for i in range(len(layers)):
        if not is_traget_block(layers[i]):
            continue
        head_groups = reorder_params["head_groups"][i]
        channel_group = reorder_params["channel_group"][i]
        mixer = get_mamba(layers[i])
        for name, m in mixer.named_modules():
            if is_calib_ops(m):
                # FIXME(HY): hardcode everything for now
                a_bits = 8
                a_clip_ratio = 1.0
                op = name.split(".")[0]
                if is_x(op):
                    observers[i][op + ":input"] = CrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                elif is_BC(op):
                    observers[i][op + ":input"] = PerSSDGroupObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        dstate=mixer.d_state,
                    )
                elif is_ssm_state(op):
                    observers[i][op + ":input"] = CachedStatesCrossHeadMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True,
                        ngroups=mixer.ngroups,
                        headdim=mixer.headdim,
                        dstate=mixer.d_state,
                        head_groups=head_groups,
                        channel_group=channel_group,
                    )
                else:
                    observers[i][op + ":input"] = PerTensorMinmaxObserver(
                        n_bits=a_bits,
                        clip_ratio=a_clip_ratio,
                        sym=True
                    )
                observers[i][op + ":output"] = PerTensorMinmaxObserver(
                    n_bits=a_bits,
                    clip_ratio=a_clip_ratio,
                    sym=True
                )
                hooks.append(
                    m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
                )
    # add observer hook for lm_head
    observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
        n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
    hooks.append(
        model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
    )
    if calibration_dataset is None:
        logger.info("Calibrate with monology/pile-uncopyrighted")
        calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
        calibration_dataset.shuffle(seed=42)

        device = next(model.parameters()).device
        def preprocess(data, tokenizer, max_tokens, device):
            input_ids = tokenizer(data["text"], return_tensors="pt",
                    max_length=max_tokens, truncation=True).input_ids.to(device)
            return input_ids
        preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

    logger.info("Run calibration")
    for i in tqdm(range(num_samples)):
        input_ids = preprocess_fn(calibration_dataset[i])
        # prepare inference cache for getting ssm_state scales
        prompt_len = input_ids.size(1)
        inf_cache = model.allocate_inference_cache(1, prompt_len)
        lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
        inference_params = InferenceParams(
            max_seqlen=prompt_len,
            max_batch_size=1,
            seqlen_offset=0,
            key_value_memory_dict=inf_cache,
            lengths_per_sample=lengths_per_sample,
        )
        # do not set num_last_tokens because we want all activations to lm_head
        model(input_ids, inference_params=inference_params)
        # clean up the cache
        del inf_cache
    
    for h in hooks:
        h.remove()
    
    # collect in/output scaling factors for layers, num_layer + lm_head
    act_scales = [{} for _ in range(len(layers) + 1)]
    for i in range(len(layers) + 1):
        for name, observer in observers[i].items():
            scale, base = observer.get_quantization_parameters()
            # FIXME (HY): hardcode to not use base now
            act_scales[i][name] = scale

    return act_scales

@torch.no_grad()
def fuse_had_matrices(model):
    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gptq
    layers = model.backbone.layers
    for i in range(len(layers)):
        # in_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.in_proj, HadLinear):
            layers[i].mixer.in_proj.fuse_hadamard()
        # out_proj: fuse had matrices with weight matrices
        if isinstance(layers[i].mixer.out_proj, HadLinear):
            layers[i].mixer.out_proj.fuse_hadamard()
    return model

@torch.no_grad()
def apply_gptq(model, tokenizer, device,args,w_bits=4,):
    # Hardcode gptq hyper-parameters for now
    nsamples = 128
    seqlen = 1024
    bits = w_bits
    assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
    logging.info("Start Quantized Linear Layers with GPTQ")
    logging.info("* Number of samples: %d" % nsamples)
    logging.info("* Sequence length: %d" % seqlen)
    logging.info("* Target bit-width for weights: %d" % bits)
    logging.info("Build calibration loader for GPTQ")
    #build dataloader
    dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
    layers = model.backbone.layers
    model.backbone.embedding = model.backbone.embedding.to(device)
    layers[0] = layers[0].to(device)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    
    residual = torch.zeros(
        (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
    )    

    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module  
        def forward(self, inp, res = None, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass

    # the hook to collect inputs for in_proj, out_proj, and lm_head
    def add_batch(module, inp, out, gptq):
        gptq.add_batch(inp[0].data, out.data)

    layers[0] = layers[0].module # remove Catcher
    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    for i in tqdm(range(len(layers))):
        # get layer
        layer = layers[i].to(device)
        #print(f"BeforeQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight.norm())
        # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
        #print(f"BeforeQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight)
        # torch.set_printoptions(profile='default')
        if args and args.lowrank:
            W = layer.mixer.out_proj.weight.data.clone()                                        #W동일
            branch  = LowRankBranch(W.shape[1], W.shape[0],
                                    rank=args.lr_rank, weight=W)
            layer.mixer.lowrank_branch = branch             # 훅용
            # layer.mixer.SVDweight = (W - branch.get_effective_weight())  # ← R
            #print("In GPTQ, W",layer.mixer.out_proj.weight)
            layer.mixer.out_proj.weight.data.copy_(W - branch.get_effective_weight())  # ← R
            #print("In GPTQ, R",layer.mixer.out_proj.weight)
            del W
        # create GPTQ objects for in_proj and out_proj
        gptq = {
            "in_proj": GPTQ(layer.mixer.in_proj),
            "out_proj": GPTQ(layer.mixer.out_proj),
        }
        handles = [
            layer.mixer.in_proj.register_forward_hook(partial(add_batch, gptq=gptq["in_proj"])),
            layer.mixer.out_proj.register_forward_hook(partial(add_batch, gptq=gptq["out_proj"]))
        ]


        for j in range(nsamples):
            layer(
                inps[j].unsqueeze(0), 
                residual=residual[j].unsqueeze(0)
            )
        for h in handles:
            h.remove()   
                                          
        
        # start running GPTQ
        for name in gptq.keys():
            logging.debug(f"Performing GPTQ on layer.{i}.mixer.{name} with {bits} bits")
            gptq[name].fasterquant(
                percdamp=0.01, group_size=128, w_bits=bits
            )
            gptq[name].free()
        del gptq
        
        # collect the outputs for the next layer
        for j in range(nsamples):
            inps[j], residual[j] = layer(inps[j].unsqueeze(0), residual=residual[j].unsqueeze(0))
        
        # garbage collection and clean cache
        layers[i] = layer.cpu()
        #print(f"AFter Quantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight.norm())
        # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
        #print(f"AfterQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight)
        # torch.set_printoptions(profile='default')
        del layer
        torch.cuda.empty_cache()
        gc.collect()

    model = model.to("cpu") # move model to cpu to save memory
    model.lm_head = model.lm_head.to(device)
    model.backbone.norm_f = model.backbone.norm_f.to(device)
    logging.info("Quantizing lm_head with GPTQ")
    gptq_lm_head = GPTQ(model.lm_head)
    handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))
    
    assert model.backbone.fused_add_norm, "Only support fused_add_norm=True for now"
    #Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L202
    final_hidden_states = layer_norm_fn(
        x=inps,
        weight=model.backbone.norm_f.weight,
        bias=model.backbone.norm_f.bias,
        eps=model.backbone.norm_f.eps,
        residual=residual,
        prenorm=False,
        residual_in_fp32=model.backbone.residual_in_fp32,
        is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
    )
    
    for j in range(nsamples):
        model.lm_head(final_hidden_states[j].unsqueeze(0))

    handle.remove()
    # compute with fp16 to save memory
    gptq_lm_head.fasterquant(
        percdamp=0.01, group_size=128, dtype=torch.float16
    )
    gptq_lm_head.free()
    del gptq_lm_head
    
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    return model


def quantize_norm_a8(block_type, norm, layer_idx, act_scales, device):
    norm = QRMSNorm.from_fp16(
        originalLayer=norm,
        output_scale=act_scales[layer_idx]["in_proj:input"].item())
    return norm.to(device)


def quantize_mixer_w8a8(block_type, mixer, layer_idx, act_scales,inp_smoothing,out_smoothing,use_had_transform, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
    W8A8Mixers = {
        "Mamba": W8A8QMamba,
        "Mamba2": W8A8QMamba2,
    }
    if block_type not in W8A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W8A8 Mixer")
    if W8A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W8A8")
    mixer = W8A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                # use_had_transform=True,
                use_lowrank=use_lowrank,
                use_had_transform=use_had_transform,  
                inp_smoothing=inp_smoothing,
                lr_rank=lr_rank,
                out_smoothing=out_smoothing,
                
                use_mixed_pre=use_mixed_pre,
                use_squeeze=use_squeeze)
    return mixer.to(device)

def quantize_mixer_w4a16(block_type, mixer, layer_idx, act_scales, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
    W4A16Mixers = {
        "Mamba": W4A16QMamba,
        "Mamba2": W4A16QMamba2,
    }
    if block_type not in W4A16Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A16 Mixer")
    if W4A16Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A16")
    mixer = W4A16Mixers[block_type].from_fp16(
                originalLayer=mixer, 
                use_had_transform=True,
                use_lowrank=use_lowrank,   
                lr_rank=lr_rank,
                use_mixed_pre=use_mixed_pre,
                use_squeeze=use_squeeze)
    return mixer.to(device)

def quantize_mixer_w4a8(block_type, mixer, layer_idx, act_scales,out_smoothing,inp_smoothing,use_had_transform, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
    W4A8Mixers = {
        "Mamba": W4A8QMamba,
        "Mamba2": W4A8QMamba2,
    }
    if block_type not in W4A8Mixers.keys():
        raise ValueError(f"Not find {block_type} in W4A8 Mixer")
    if W4A8Mixers[block_type] is None:
        raise ValueError(f"Not support {block_type} with W4A8")
    #print("[DEBUG] quantize_mixer_w4a8 use_lowrank =", use_lowrank)
    mixer = W4A8Mixers[block_type].from_fp16(
                originalLayer=mixer,
                act_scales=act_scales[layer_idx],
                # use_had_transform=True,
                use_lowrank=use_lowrank,   
                lr_rank=lr_rank,
                use_mixed_pre=use_mixed_pre,
                out_smoothing=out_smoothing,
                inp_smoothing=inp_smoothing,
                use_had_transform=use_had_transform,
                use_squeeze=use_squeeze)
    return mixer.to(device)

def get_quantize_block_fn(act_scales, w_bits, a_bits, device,inp_smoothing, out_smoothing, use_had_transform):
    if w_bits == 4 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w4a8, act_scales=act_scales, device=device,inp_smoothing=inp_smoothing, out_smoothing=out_smoothing, use_had_transform=use_had_transform)
    elif w_bits == 4 and a_bits == 16:
        quantize_norm_fn = lambda block_type, norm, layer_idx: norm # just return the original layer
        quantize_mixer_fn = partial(quantize_mixer_w4a16, act_scales=act_scales, device=device)
    elif w_bits == 8 and a_bits == 8:
        quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
        quantize_mixer_fn = partial(quantize_mixer_w8a8, act_scales=act_scales, device=device,inp_smoothing=inp_smoothing, out_smoothing=out_smoothing, use_had_transform=use_had_transform)
    else:
        raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    return quantize_norm_fn, quantize_mixer_fn
    
@torch.no_grad()
def quantize_fp16_model(model, model_type, act_scales, device, w_bits=4, a_bits=8, quantize_embedding=True, quantize_lm_head=True,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False, inp_smoothing=False,out_smoothing=False, use_had_transform=False):
    assert w_bits in [1, 4, 8], "Only support 4 or 8 bits weights for now"
    assert a_bits in [1, 8, 16], "Only support 8 or 16 bits activations for now"
    quantize_norm_fn, quantize_mixer_fn = get_quantize_block_fn(act_scales, w_bits, a_bits, device, inp_smoothing, out_smoothing, use_had_transform)
    model.config.use_cache = False
    if model_type == "mamba":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                # replace with fused RMSNorm
                #print(f"[DEBUG] layer {i} use_lowrank in quantize_fp16_model:", use_lowrank)
                layers[i].fused_add_norm = True
                layers[i].norm = quantize_norm_fn(
                    block_type="Mamba",
                    norm=layers[i].norm,
                    layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba", 
                    mixer=layers[i].mixer,
                    layer_idx=i,
                    
                    )
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    elif model_type == "mamba2":
        if quantize_embedding:
            # replace embedding layer
            logging.info(f'Applying quantized embedding')
            if w_bits == 4:
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 8:
                model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
            elif w_bits == 1:                                                                       ##일단 88취급
                model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
                
            else:
                raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
            gc.collect()
            torch.cuda.empty_cache()
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
                if not inp_smoothing: ##DO## for debugging
                    layers[i].norm = quantize_norm_fn(
                            block_type="Mamba2",
                            norm=layers[i].norm,
                            layer_idx=i)
                layers[i].mixer = quantize_mixer_fn(
                    block_type="Mamba2", 
                    mixer=layers[i].mixer,
                    layer_idx=i,
                    use_lowrank=use_lowrank,   # 새 인자
                    lr_rank=lr_rank,
                    use_mixed_pre=use_mixed_pre,
                    use_squeeze=use_squeeze)
                # garbage collection and clean cache
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    
    if quantize_lm_head:
        logging.info(f'Applying quantized lm_head')
        # replace lm_head and norm_f with quantized version
        if w_bits == 4 and a_bits == 16:
            # do nothing to w4a16 norm_f
            model.lm_head = W4A16B16O16Linear.from_fp16(model.lm_head)
        elif w_bits == 4 and a_bits == 8:
            # model.backbone.norm_f = FusedRMSNorm.from_fp16(
            model.backbone.norm_f = QRMSNorm.from_fp16(
                model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
            model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
        elif w_bits == 8 and a_bits == 8:
            # model.backbone.norm_f = FusedRMSNorm.from_fp16(
            model.backbone.norm_f = QRMSNorm.from_fp16(
                model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
            model.lm_head = W8A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"].item())
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    
    gc.collect()
    torch.cuda.empty_cache()
    return model

def quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=4, 
                                   layer_wise_hybrid_config=None #this is expected to be a list
                                   ):
    assert w_bits in [4], "Only support 4 bits weights for now"
    a_bits = [8, 16]
    
    logging.info(f"Quantizing model with w{w_bits} and a{a_bits}. HybridBlock will be create.")
    
    # for each a_bits get the correcponding quantization function
    quant_function_pairs = {}
    for a in a_bits:
        quant_function_pairs[f'W{w_bits}A{a}'] = get_quantize_block_fn(act_scales, w_bits, a, device) # quantize_norm_fn, quantize_mixer_fn
    
    model.config.use_cache = False
    if model_type == "mamba2":
        # replace embedding layer
        logging.info(f'Applying quantized embedding')
        if w_bits == 4:
            model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
        elif w_bits == 8:
            model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
        else:
            raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
        # replace layers
        logging.info(f'Applying quantized layers')
        layers = model.backbone.layers
        for i in tqdm(range(len(layers))):
            if isinstance(layers[i], Block):
                layers[i].fused_add_norm = True
                
                mixers = {}
                norms = {}
                
                if layer_wise_hybrid_config:
                    #Case 1: bitwidth of each layer is specified, each layers only create the block/norm with the specified bitwidth
                    bit_config = layer_wise_hybrid_config[i]
                    (quantize_norm_fn, quantize_mixer_fn) = quant_function_pairs[bit_config]
                    mixers[bit_config] = quantize_mixer_fn(
                        block_type="Mamba2", 
                        mixer=layers[i].mixer,
                        layer_idx=i)
                    norms[bit_config] = quantize_norm_fn(
                        block_type="Mamba2",
                        norm=layers[i].norm,
                        layer_idx=i)     
                else:
                    #Case 2: bitwidth of each layer is not specified, each layers create the block/norm with all possible bitwidth
                    for bits, (quantize_norm_fn, quantize_mixer_fn) in quant_function_pairs.items():
                        mixers[f"W{w_bits}A{bits}"] = quantize_mixer_fn(
                            block_type="Mamba2", 
                            mixer=layers[i].mixer,
                            layer_idx=i)
                        norms[f"W{w_bits}A{bits}"] = quantize_norm_fn(
                            block_type="Mamba2",
                            norm=layers[i].norm,
                            layer_idx=i)          
                            
                layers[i] = HybridQuambaBlock.from_block_and_mixer_norm_dict(
                    block=layers[i],
                    mixers_dict=mixers,
                    norms_dict=norms,
                ).to(device)
                layers[i].set_mixer(next(iter(mixers))) #default
                gc.collect()
                torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2' for hybrid mode")
    
    logging.info(f'Applying quantized lm_head')
    
    #FIXME(brian1009): Hard-fix for now, but we may need to make it configurable as well.
    model.backbone.norm_f = QRMSNorm.from_fp16(
        model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
    model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
    
    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_model_size(model, model_name, w_bits, a_bits):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_mb = (param_size + buffer_size) / 1024**2
    logging.info(f'W{w_bits}A{a_bits} {model_name} size: {model_mb:.3f} MB')


def quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None,):
    # restore the quantized model from pretrained_dir
    if args.pretrained_dir:
        # change the name to lookup the model in the pretrained_dir
        model_name = args.model.lower().split('/')[-1]

        # already loaded the quantized model in build_mamba_and_tokenizer in utils.py
        if model_name.startswith("quamba"): # ut-enyac/quamba or ut-enyac/quamba2
            get_model_size(model, args.model, args.w_bits, args.a_bits)
            return model

        # load quantied model in args.pretrained_dir to replace fp16 mamba
        # This will require much more memory, since we will have
        # fp16 mamba, qumaba, and quamba weight in the memory at the same time
        if model_name.startswith("mamba"): # mamba or mamba2
            model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
            if args.hybrid_blocks: 
                model_name = model_name + f"-w{args.w_bits}aX"
                if args.hybrid_blocks_config:
                    config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
                    model_name = model_name + f"-{config_name}"
            else:
                model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
        else:
            logging.warning(f"Unsupported model {args.model} in ut-enyac/ model registry")
        # load the quantized model if it exists
        if os.path.isdir(quantized_model_path):
            logging.info(f"Loading quantized model from {quantized_model_path}")
            model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
            get_model_size(model, args.model, args.w_bits, args.a_bits)
            return model
        else:
            logging.warning(f"{quantized_model_path} does not exist.")
            logging.warning("Runing calibration and quantization from scratch")
    if args.apply_smoothing:
        args.apply_inp_smoothing, args.apply_out_smoothing = True, True
        args.smoothing_inp_alpha, args.smoothing_out_alpha = args.smoothing_alpha, args.smoothing_alpha
    # replace the mamba blocks with simple blocks to get the scaling factors
    # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
    model = configure_model(model, model_type, use_had_transform=args.apply_hadamard, inp_smoothing=args.apply_inp_smoothing,out_smoothing=args.apply_out_smoothing)   ##DO## # W4A16 needs had_transform as well #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """아다마르 변환 일단 적용 X"""
    "smoothing point" ##DO##
        ###########################################################가중치 시각화##################################################
    # import torch, matplotlib.pyplot as plt, numpy as np
    # from pathlib import Path
    # def plot_row_stats(
    #     W: torch.Tensor,
    #     name="W",
    #     save_dir="row_plots",
    #     y_lim=None,                          # ← 기본값 None = 자동 스케일
    #     colors=("forestgreen", "goldenrod", "maroon"),
    #     dpi=140):

    #     W = W.detach().float().cpu()
    #     med  = W.median(dim=0).values
    #     p99  = torch.quantile(W, .99, dim=0)
    #     vmax = W.max(dim=0).values
    #     xs   = np.arange(W.shape[1])

    #     fig, ax = plt.subplots(figsize=(9,3), dpi=dpi)
    #     ax.fill_between(xs, 0, med,  color=colors[0], alpha=.8, label="median")
    #     ax.fill_between(xs, med, p99, color=colors[1], alpha=.8, label="99 %")
    #     ax.fill_between(xs, p99, vmax,color=colors[2], alpha=.8, label="max")

    #     ax.set(xlabel="col index",
    #         ylabel="Weight value",
    #         title=f"{name} — per-col stats")
    #     if y_lim is not None:                   # ← 자동/고정 선택
    #         ax.set_ylim(*y_lim)
    #     ax.margins(x=0)
    #     ax.legend(fontsize=8, framealpha=.9)
    #     fig.tight_layout()

    #     Path(save_dir).mkdir(exist_ok=True, parents=True)
    #     fname = Path(save_dir) / f"{name.lower()}_row_stat_smoothba.png"
    #     fig.savefig(fname); plt.close(fig)
    #     print("✓ saved →", fname)
###############################################################################가중치시각화#############################################################
    #W_smooth_before = model.backbone.layers[-1].mixer.out_proj.weight
    #print("BeforeSmooth",model.backbone.layers[-1].mixer.out_proj.weight)
    #plot_row_stats(W_smooth_before, "W_smooth_before")
    for i, blk in enumerate(model.backbone.layers):
        W_blk = blk.mixer.out_proj.weight
        #print(f"Layer {i} - before had(smooth) weight norm: {W_blk.norm()}")
    if args.apply_inp_smoothing or args.apply_out_smoothing:
        logging.info(f"Start doing smoothing")
        smooth_mamba(model, tokenizer, num_samples=5 if args.testing else 512, inp_smoothing = args.apply_inp_smoothing, out_smoothing=args.apply_out_smoothing,
                     inp_alpha=args.smoothing_inp_alpha, out_alpha=args.smoothing_out_alpha)
        #print("AfterSmooth",model.backbone.layers[-1].mixer.out_proj.weight)
        #W_smooth_after = model.backbone.layers[-1].mixer.out_proj.weight
        #plot_row_stats(W_smooth_after,      "W_smooth_after")
    # replace the mamba blocks with simple blocks to get the scaling factors
    # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
    ##############################################original code#############################################################
    #model = configure_model(model, model_type, use_had_transform=True) # W4A16 needs had_transform as well
    #####################################################################################################
    logging.info(f"Target bit-width W{args.w_bits}A{args.a_bits}")
    if args.a_bits == 8:
        # Run calibration to get scale and reorder params
        if args.group_heads:
            logging.info(f"Reordering weights and activations for head grouping")
            reorder_params = get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512)
            reorder_mamba(model, reorder_params)
            # collect 8-bit activation scales
            act_scales = run_quamba2_calibration(model, model_type, tokenizer, reorder_params,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
        else:
            # collect 8-bit activation scales
            act_scales, fp_means = run_quamba_calibration(model, model_type, tokenizer,
                                                num_samples=args.calib_data_num,
                                                seq_len=args.calib_seqlen,
                                                calibration_dataset=calibration_dataset,
                                                preprocess_fn=calib_preprocess_fn)
    elif args.a_bits == 16:
        # not doing anything for activations
        act_scales = {}
        if args.compensation :
            logging.info(f"Collecting fp means for compensation")
            _, fp_means = run_quamba_calibration(model, model_type, tokenizer,
                                        num_samples=args.calib_data_num,
                                        seq_len=args.calib_seqlen,
                                        calibration_dataset=calibration_dataset,
                                        preprocess_fn=calib_preprocess_fn)
        if args.group_heads:
            logging.info(f"Activation bit-width is set to 16, skip weights reordering and head grouping")
    else:
        raise ValueError(f"Unsupported activation bit-width: {args.a_bits}, try --a_bits 8 or --a_bits 16?")
    

    # fuse the had matrices with the weight matrices in linear layers.
    # Do this after reordering and before applying gpt
    model = fuse_had_matrices(model) 
    #print("AfterSMOOTH FUSE ",model.backbone.layers[-1].mixer.out_proj.weight)
    # if args.lowrank and not args.apply_gptq:
    #     for blk in model.backbone.layers:
    #         lr_rank = args.lr_rank

    #         # ── 1. GPU weight → W_fp16
    #         W_fp16 = blk.mixer.out_proj.weight.data          # (GPU - fp16)

    #         # ── 2. GPU에서 branch 만들고 잔차 R 계산
    #         branch = LowRankBranch(
    #             in_features=W_fp16.shape[1],
    #             out_features=W_fp16.shape[0],
    #             rank=lr_rank,
    #             weight=W_fp16,                               # <- GPU tensor
    #         )

    #         R = W_fp16 - branch.get_effective_weight()       # R ⟵ GPU 계산
    #         blk.mixer.out_proj.weight.data.copy_(R)          # 잔차로 덮어쓰기

    #         # ── 3. branch 는 CPU 로 옮겨서 저장, GPU 메모리 해제
    #         blk.mixer.lowrank_branch = branch.cpu()        # branch(fp16, CPU)
    #         del R                                            # 덜 필요하지만 확실히
    #         torch.cuda.empty_cache()
    # Apply GPTQ to quantize linear
    #print("BeforeGPTQ",model.backbone.layers[-1].mixer.out_proj.weight)
    if args.apply_gptq:
        model = apply_gptq(model, tokenizer, device, args , w_bits=args.w_bits)
    # Replace (reordered, fused, and GPTQ quantized) modules with quantized version
    #print("AfterGPTQ",model.backbone.layers[-1].mixer.out_proj.weight)
    if args.hybrid_blocks: # create hybrid block
        if args.hybrid_blocks_config:
            logging.info(f"Loading hybrid blocks config from {args.hybrid_blocks_config}")
            with open(args.hybrid_blocks_config, "r") as file:
                hybrid_blocks_configs = json.load(file)
        else:
            hybrid_blocks_configs = None
        model = quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=args.w_bits,
                                               layer_wise_hybrid_config=hybrid_blocks_configs)
    else:
        print("[DEBUG] args.lowrank =", args.lowrank)  # <-- 여기
        model = quantize_fp16_model(
            model, model_type, act_scales, device,
            w_bits=args.w_bits, a_bits=args.a_bits,
            quantize_embedding=args.quantize_embedding,
            quantize_lm_head=args.quantize_lm_head,
            use_lowrank=args.lowrank, 
            lr_rank=args.lr_rank,
            use_mixed_pre=args.mixed_pre,
            use_squeeze=args.squeeze,
            inp_smoothing=args.apply_inp_smoothing,
            out_smoothing=args.apply_out_smoothing,
            use_had_transform=args.apply_hadamard
        )
        print("quantization complete")
        
        if args.compensation :
            model._fp_means = fp_means
            model = mamba_sequential_compensation_ssdout(
                model=model,
                device=device,
                nsamples=args.comp_sam_num,
                tokenizer=tokenizer,
                seq_len=args.calib_seqlen,
                dataloader=None,                      # 옵션 A를 쓰면 여기에 로더
                dataset=calibration_dataset,          # ← run_quamba_calibration에 쓴 것과 동일
                preprocess_fn=calib_preprocess_fn,     # ← run_quamba_calibration에 쓴 것과 동일
                comp_out_decay=args.comp_out_decay,
                comp_ssd_decay=args.comp_ssd_decay,
                layers=args.comp_layers
            )
            model = model.to(device).eval()
            torch.cuda.empty_cache()
        
        
        
    # store the state_dict if not quamba
    model_name = args.model.lower().split('/')[-1]
    if args.pretrained_dir is not None and not model_name.startswith("quamba"):
        if not args.hybrid_blocks:
            # change the name to store the model in the pretrained_dir
            model_name = args.model.lower().split('/')[-1]
            model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
            model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"

            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            # we slightly hack the api: we use MambaLMHeadModel instead of QuambaLMHeadModel to store the model here
            model.config.ssm_cfg['layer'] = model.backbone.layers[0].mixer.__class__.__name__
            model.config.norm_cfg = {"norm": model.backbone.layers[0].norm.__class__.__name__}
            model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
            model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            if hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            model.save_pretrained(quantized_model_path)
            logging.info(f"The quantized model is stored at {quantized_model_path}")
        else:
            model_name = args.model.lower().split('/')[-1]
            model_name = model_name.replace("mamba", "quamba")
            model_name = model_name + f"-w{args.w_bits}aX"
            if args.hybrid_blocks_config:
                config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
                model_name = model_name + f"-{config_name}"
            quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            
            ssm_layer_infos = []
            norm_infos = []
            
            for i in range(len(model.backbone.layers)):
                mixer_norms_info = model.backbone.layers[i].get_class_info_dict_mixers_norms()
                ssm_layer_infos.append(mixer_norms_info["mixers"])
                norm_infos.append(mixer_norms_info["norms"])
            
            mixer_norms_info = model.backbone.layers[0].get_class_info_dict_mixers_norms()
            model.config.ssm_cfg['layer'] = ssm_layer_infos
            model.config.norm_cfg = {"norm": norm_infos}
            model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
            model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
            # We apply Hadamard transforms so we cannot tie embeddings and lm_head
            model.config.tie_embeddings = False # no used in QuambaLMHeadModel
            if hasattr(model.config, "use_cache"):
                delattr(model.config, "use_cache")
            model.config.is_hybrid = True
            model.save_pretrained(quantized_model_path)
            logging.info(f"The quantized model is stored at {quantized_model_path}")
            
        # store tokenizer for mamba2-8b
        if "mamba2-8b" in args.model:
            # model.save_pretrained should already create the saved dir
            saved_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            tokenizer.save(saved_dir)
            logging.info(f"Tokenizer is stored at {saved_dir}")
    # quantized model
    get_model_size(model, args.model, args.w_bits, args.a_bits)
    return model.to(device)





@torch.no_grad()
def mamba_sequential_compensation_ssdout(
    model,
    device,
    nsamples,
    tokenizer,
    seq_len=2048,
    dataloader=None,
    dataset=None,
    preprocess_fn=None,
    # comp_in_decay 제거됨 (in_proj 보상 미사용)
    comp_out_decay=0.1,
    comp_ssd_decay=0.1,     # ssd 보상 감쇠
    layers=None,            # 선택 레이어만 보상
):
    import torch
    from torch import nn
    from tqdm import tqdm

    def _batch_sum_and_count(y: torch.Tensor):
        if y.dim() == 3: return y.sum(dim=(0,1)).float(), y.shape[0]*y.shape[1]
        if y.dim() == 2: return y.sum(dim=0).float(), y.shape[0]
        raise RuntimeError(f"Unexpected shape {tuple(y.shape)}")

    model.eval()
    blocks = model.backbone.layers
    fp_means = getattr(model, "_fp_means", None)
    assert isinstance(fp_means, dict)

    # --- 캡처 준비(입력 x만) ---
    model.backbone.embedding = model.backbone.embedding.to(device)
    blocks[0] = blocks[0].to(device)

    if dataloader is None and (dataset is None or preprocess_fn is None):
        from datasets import load_dataset
        dataset = load_dataset("monology/pile-uncopyrighted",
                               data_files="val.jsonl.zst", split="train")
        def _preprocess(data, tokenizer, max_tokens, device):
            return tokenizer(data["text"], return_tensors="pt",
                             truncation=True, max_length=max_tokens
                            ).input_ids.to(device)
        from functools import partial
        preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
                                max_tokens=seq_len, device=device)

    inps_list = []
    class Catcher(nn.Module):
        def __init__(self, mod): super().__init__(); self.mod = mod
        def forward(self, x, *args, **kwargs):
            inps_list.append(x.squeeze(0).detach() if (x.dim()==3 and x.size(0)==1) else x.detach())
            raise ValueError
    blocks[0] = Catcher(blocks[0])

    if dataloader is not None:
        it = iter(dataloader)
        for _ in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
            batch = next(it)
            try:
                xb = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(xb.to(device))
            except ValueError:
                pass
    else:
        for i in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
            x = preprocess_fn(dataset[i])
            try: model(x)
            except ValueError: pass

    blocks[0] = blocks[0].mod
    blocks[0] = blocks[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    torch.cuda.empty_cache()
    assert len(inps_list) == nsamples

    # residual 초기화(모델이 residual을 반환하는 인터페이스일 경우 사용)
    res_in_list = [None] * nsamples

    # --- 레이어 선택 집합 ---
    L = len(model.backbone.layers)
    def _norm_idxs(idxs):
        s=set()
        for k in idxs:
            if k<0: k=L+k
            if 0<=k<L: s.add(int(k))
        return s

    if layers is None:
        target_layers = set(range(L))
    else:
        target_layers = _norm_idxs(list(layers) if not isinstance(layers, (list, tuple)) else layers)

    # --- 레이어 순차 보상 ---
    for li in tqdm(range(L), desc="[Layer] Sequential compensation", leave=True):
        layer = blocks[li].to(device)
        mix = layer.mixer
        mix.compensation = True

        # 대상이 아니면 그냥 전파
        if li not in target_layers:
            next_inps, next_res = [], []
            for j in range(nsamples):
                xj = inps_list[j];  xj = xj.unsqueeze(0) if xj.dim()==2 else xj
                yj, resj = layer(xj, residual=res_in_list[j])
                next_inps.append(yj.squeeze(0).detach())
                next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
            blocks[li] = layer.cpu(); torch.cuda.empty_cache()
            inps_list, res_in_list = next_inps, next_res
            continue

        # ===== 보상 버퍼 준비 =====
        # in_proj 보상은 사용하지 않으므로 comp_in은 0 버퍼로 고정
        # if getattr(mix, "comp_in", None) is None:
        #     mix.register_buffer("comp_in", torch.zeros(
        #         mix.in_proj.out_features, dtype=torch.float16, device=device))
        # else:
        #     mix.comp_in.zero_()

        if getattr(mix, "ssd_comp", None) is None:
            mix.register_buffer("ssd_comp", torch.zeros(
                mix.d_inner, dtype=torch.float16, device=device))
        if getattr(mix, "comp_out", None) is None:
            mix.register_buffer("comp_out", torch.zeros(
                mix.out_proj.out_features, dtype=torch.float16, device=device))

        # fp 기준 평균
        m_fp_ssd  = fp_means.get(f"L{li}/ssd_out",      None)
        m_fp_out  = fp_means.get(f"L{li}/out_proj_out", None)
        if m_fp_ssd is not None: m_fp_ssd = m_fp_ssd.to(device=device, dtype=torch.float32)
        if m_fp_out is not None: m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)

        # 모든 보상 초기화
        # mix.comp_in.zero_()
        mix.ssd_comp.zero_()
        mix.comp_out.zero_()

        # =========================
        # Phase B: ssd_comp 추정
        # (comp_in=0, ssd_comp/off, comp_out/off)
        # =========================
        ssd_sum=None; ssd_cnt=0
        if hasattr(mix, "ssd_out_act") and mix.ssd_out_act is not None and (m_fp_ssd is not None):
            def ssd_hook(_m,_i,out):
                nonlocal ssd_sum, ssd_cnt
                s, n = _batch_sum_and_count(out)
                ssd_sum = s if ssd_sum is None else (ssd_sum+s)
                ssd_cnt += int(n)
            hB = mix.ssd_out_act.register_forward_hook(ssd_hook)
            for j in tqdm(range(nsamples), desc=f"[L{li}] Phase B: ssd_out", leave=False):
                xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
                _ = layer(xj, residual=res_in_list[j], comp_calib=True)
            hB.remove()
            if ssd_cnt > 0:
                q_mean_ssd = torch.where(torch.isfinite(ssd_sum/ssd_cnt), ssd_sum/ssd_cnt, torch.zeros_like(ssd_sum))
                mix.ssd_comp.copy_((m_fp_ssd - q_mean_ssd).to(mix.ssd_comp.dtype) * comp_ssd_decay)

        # =========================
        # Phase C: comp_out 추정
        # (comp_in=0, ssd_comp ON, comp_out/off)
        # =========================
        out_sum=None; out_cnt=0
        def out_hook(_m,_i,out):
            nonlocal out_sum,out_cnt
            s, n = _batch_sum_and_count(out)
            out_sum = s if out_sum is None else (out_sum+s)
            out_cnt += int(n)
        hC = mix.out_proj.register_forward_hook(out_hook)
        for j in tqdm(range(nsamples), desc=f"[L{li}] Phase C: out_proj", leave=False):
            xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
            _ = layer(xj, residual=res_in_list[j], comp_calib=True)
        hC.remove()
        if (m_fp_out is not None) and (out_cnt > 0):
            q_mean_out = torch.where(torch.isfinite(out_sum/out_cnt), out_sum/out_cnt, torch.zeros_like(out_sum))
            mix.comp_out.copy_((m_fp_out - q_mean_out).to(mix.comp_out.dtype) * comp_out_decay)

        # =========================
        # Phase D: propagate (ssd_comp + comp_out on, comp_in=0)
        # =========================
        next_inps, next_res = [], []
        for j in tqdm(range(nsamples), desc=f"[L{li}] Phase D: propagate", leave=False):
            xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
            yj, resj = layer(xj, residual=res_in_list[j], comp_calib=True)
            next_inps.append(yj.squeeze(0).detach())
            next_res.append(resj.detach() if torch.is_tensor(resj) else resj)

        blocks[li] = layer.cpu(); torch.cuda.empty_cache()
        inps_list, res_in_list = next_inps, next_res

    model._fp_means = None
    return model





# import os
# import gc
# import copy
# import logging
# from tqdm import tqdm
# from functools import partial
# import json

# import torch
# import torch.nn as nn
# from datasets import load_dataset

# from mamba_ssm.modules.block import Block
# from mamba_ssm.modules.mamba2 import Mamba2
# from mamba_ssm.modules.mamba_simple import Mamba
# from mamba_ssm.utils.generation import InferenceParams
# from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm

# from quamba.quamba_mixer_seq import QuambaLMHeadModel
# from .qEmbedding import W4O16Embedding, W8O16Embedding
# from .qLinearLayer import HadLinear, W4A16B16O16Linear, W4A8B16O16Linear, W8A8B16O16Linear
# from .qActLayer import ActIdentity
# from .qMamba2 import Mamba2Simple, W4A8QMamba2, W4A16QMamba2, W8A8QMamba2
# from .qMambaLayer import MambaSimple, W4A8QMamba, W4A16QMamba, W8A8QMamba
# from .qHadamard import Hadamard
# from .qBlock import HybridQuambaBlock
# # from .fusedNorm import FusedRMSNorm
# from .qNorm import QRMSNorm
# from .observer import PerTensorMinmaxObserver, PerTensorPercentileObserver, _ChannelMeanCollector
# from .observer import PerSSDGroupObserver, CrossHeadMinmaxObserver
# from .observer import CachedStatesCrossHeadMinmaxObserver
# from .observer import ChunkCollector
# from .gptq_utils import GPTQ
# from .reorder_utils import get_reorder_params, reorder_mamba
# from .hadamard_utils import had_transform
# from .data_loaders import get_loaders
# from .smooth_quant_utils import smooth_mamba
# from lowrank import LowRankBranch  


# logger = logging.getLogger(__name__)

# @torch.no_grad()
# def fuse_ln_linear(norm, linear) -> None:
#     """
#     fuse the layernorm weight to the adjacent linear layer.
#     """
#     linear_dtype = linear.weight.dtype

#     # Calculating new weight and bias
#     W_ = linear.weight.data.double()
#     linear.weight.data = (W_ * norm.weight.double()).to(linear_dtype)  
#     if hasattr(norm, 'bias') and norm.bias is not None:
#         if linear.bias is None:
#             linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float32))
#         linear.bias.data = linear.bias.data.double() + torch.matmul(W_, norm.bias.to(torch.float32))
#         linear.bias.data = linear.bias.data.to(linear_dtype)
#     # Reset the learnable weight in RMSNorm to 1
#     norm.weight.data = torch.ones_like(norm.weight).to(norm.weight.dtype) # Reset the weight to 1
# import time
# from contextlib import contextmanager
# @contextmanager
# def _time_block(label: str, device=None, sync=True):
#             """CUDA면 cudaEvent로, 아니면 perf_counter로 측정"""
#             use_cuda = device is not None and torch.cuda.is_available()
#             if use_cuda:
#                 if sync:
#                     torch.cuda.synchronize(device)
#                 start = torch.cuda.Event(enable_timing=True)
#                 end   = torch.cuda.Event(enable_timing=True)
#                 start.record()
#                 try:
#                     yield
#                 finally:
#                     end.record()
#                     if sync:
#                         torch.cuda.synchronize(device)
#                     ms = start.elapsed_time(end)
#                     print(f"[TIME] {label:<24} {ms:.3f} ms")
#             else:
#                 t0 = time.perf_counter()
#                 try:
#                     yield
#                 finally:
#                     t1 = time.perf_counter()
#                     print(f"[TIME] {label:<24} {(t1 - t0)*1000:.3f} ms")
# @torch.no_grad()
# def configure_model(model, model_type, use_had_transform=True, inp_smoothing=False,out_smoothing=False):
#     device = next(model.parameters()).device
#     if model_type == "mamba":
#         # process embedding and lm_head
#         if use_had_transform:            
#             # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
#             lm_head_clone = model.lm_head.weight.data.clone()
#             # transform embedding first
#             model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) 
#             # do layernorm fusion to lm_head and then transform
#             model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
#             model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
#             model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
#             torch.cuda.empty_cache()
#         layers = model.backbone.layers
#         for i in range(len(layers)):
#             if isinstance(layers[i], Block):
#                 # fuse norm to in_proj first
#                 fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)
#                 # use simplied mamba block to get the scaling factors
#                 # from linear layers without pain
#                 m = MambaSimple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform).to(device)
#                 layers[i].mixer = m
#                 torch.cuda.empty_cache()
#     elif model_type == "mamba2":
#         # process embedding and lm_head
#         if use_had_transform:            
#             # Sometimes, lm_head is tied to embedding, we make a clone for lm_head first
#             lm_head_clone = model.lm_head.weight.data.clone()
#             # transform embedding first
#             model.backbone.embedding.weight.data = had_transform(model.backbone.embedding.weight.data) #입력 임베딩 회전
#             # do layernorm fusion to lm_head and then transform
#             model.lm_head.weight = torch.nn.Parameter(lm_head_clone * model.backbone.norm_f.weight.view(1, -1)).to("cuda") # must re-initialize it with nn.Parameter to untie lm_head and embedding, otherwise, it will not work
#             ##LM-Head 와 RMSNorm 합치기
#             model.backbone.norm_f.weight.data = torch.ones_like(model.backbone.norm_f.weight)
#             ##RMSNorm 합쳤으니까 1로만들기
#             model.lm_head.weight.data = had_transform(model.lm_head.weight.data)
#             ##다시한번 회전
#             torch.cuda.empty_cache()
#         # process layers
        
#         layers = model.backbone.layers
#         #######################################
#         # print("layers=",layers)              
#         # print("lenlayers=",len(layers))   64, 2.7B -> layers=Blocks
#         # print("mdoel=,",model)
#         # for i, blk in enumerate(model.backbone.layers):        # 64개 Block 순회
#         #     n_param = sum(p.numel() for p in blk.parameters())
#         #     print(f"{i:02d} │ {blk.__class__.__name__:<10} │ {n_param/1e6:7.2f} M params")
#         # for name, module in model.named_modules():
#         #     print(f"{name:40s}  →  {module.__class__.__name__}")

#         ######################################
#         for i in range(len(layers)):
#             if isinstance(layers[i], Block):
#                 # fuse norm to in_proj first
#                 fuse_ln_linear(layers[i].norm, layers[i].mixer.in_proj)  ###RMSNorm 을 inprj에 합치기
#                 # use simplied mamba block to get the scaling factors
#                 # from linear layers without pain
#                 ##########################################################LOWRANK 적용 유력 시점: RMS만 합쳐진 상태 ########################################
#                 m = Mamba2Simple(originalLayer=layers[i].mixer, use_had_transform=use_had_transform, inp_smoothing=inp_smoothing,out_smoothing=out_smoothing).to(device)
#                 layers[i].mixer = m
#                 torch.cuda.empty_cache()
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
#     model.config.use_cache = False
#     with _time_block("eval_latency!!!!!!!", device=device):
#         model.eval()

#     # print("mdoel=,",model)
#     return model

# ######################    Mabma2  래핑 ############################
# @torch.no_grad()
# def run_quamba_calibration(
#         model, model_type, tokenizer, num_samples=512, seq_len=2048,
#         calibration_dataset=None, preprocess_fn=None
#     ):

#     if model_type == "mamba":
#         layers = model.backbone.layers
#         is_traget_block = lambda block: isinstance(block, Block)
#         get_mamba = lambda block: block.mixer
#         is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
#         is_x = lambda op: op == "x_proj"
#         is_ssm_state = lambda op: op == "ssm_state_act"
#         percentile_alpha=0.9995 # for smaller model like 130m, use 0.99999
#     elif model_type == "mamba2":
#         layers = model.backbone.layers
#         is_traget_block = lambda block: isinstance(block, Block)
#         get_mamba = lambda block: block.mixer
#         is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
#         is_x = lambda op: op == "x_conv_out"
#         is_ssm_state = lambda op: op == "ssm_state_act"
#         percentile_alpha=0.9995  # for smaller model like 130m, use 0.99999
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    
#     # register min/max observers, num_layer + lm_head
#     observers = [{} for _ in range(len(layers) + 1)]
    
#     def stat_hook(m, inputs, outputs, op, block_idx):
#         # register the new information to observer
#         if isinstance(inputs, tuple):
#             inputs = inputs[0]
#         observers[block_idx][op + ":input"].update(inputs.clone().detach())

#         if isinstance(outputs, tuple):
#             outputs = outputs[0]
#         observers[block_idx][op + ":output"].update(outputs.clone().detach())

#     hooks = []
#     fp_proj_mean_hooks = {}     ##DO
    
#     for i in range(len(layers)):
#         if not is_traget_block(layers[i]):
#             continue
#         mixer = get_mamba(layers[i])
#         x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
#         ori_x_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
#         B_col = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
#         C_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
#         state_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)
#         cb_cali = ChunkCollector(div=127.0, device=next(mixer.parameters()).device)

    
#         ##DO
#         _in_col  = _ChannelMeanCollector()
#         _out_col = _ChannelMeanCollector()
#         h_in  = mixer.in_proj.register_forward_hook(
#             lambda _m, _i, out, col=_in_col: col.update(out)
#         )
#         h_out = mixer.out_proj.register_forward_hook(
#             lambda _m, _i, out, col=_out_col: col.update(out)
#         )
#         fp_proj_mean_hooks[f"L{i}/in_proj_out"]  = (_in_col, h_in)
#         fp_proj_mean_hooks[f"L{i}/out_proj_out"] = (_out_col, h_out)
        
#         ### SSD ###
#         _ssd_col = _ChannelMeanCollector()
#         h_ssd = mixer.ssd_out_act.register_forward_hook(
#             lambda _m, _i, out, col=_ssd_col: col.update(out)
#         )
#         fp_proj_mean_hooks[f"L{i}/ssd_out"] = (_ssd_col, h_ssd)
#         ### SSD ###

#         ##DO


#         # mixer에 달아두기 → forward에서 collect_obs로 전달됨
#         mixer._collect_obs = {"ori_x_chp": ori_x_col,"x_chp": x_col, "B_cgn": B_col,"C_cali":C_cali,"state_cali":state_cali,"cb_cali":cb_cali}
#         for name, m in mixer.named_modules():
#             if is_calib_ops(m):
#                 # FIXME(HY): hardcode everything for now
#                 a_bits = 8
#                 a_clip_ratio = 1.0
#                 op = name.split(".")[0]
#                 if is_x(op) or is_ssm_state(op):
#                     observers[i][op + ":input"] = PerTensorPercentileObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True,
#                         percentile_alpha=percentile_alpha
#                     )
#                 else:
#                     observers[i][op + ":input"] = PerTensorMinmaxObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True
#                     )
#                 observers[i][op + ":output"] = PerTensorMinmaxObserver(
#                     n_bits=a_bits,
#                     clip_ratio=a_clip_ratio,
#                     sym=True
#                 )
#                 hooks.append(
#                     m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
#                 )
#     # add observer hook for lm_head
#     observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
#         n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
#     observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
#         n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
#     hooks.append(
#         model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
#     )

#     device = next(model.parameters()).device
#     if calibration_dataset is None:
#         logger.info("Calibrate with monology/pile-uncopyrighted")
#         calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")

#         def preprocess(data, tokenizer, max_tokens, device):
#             input_ids = tokenizer(data["text"], return_tensors="pt",
#                     max_length=max_tokens, truncation=True).input_ids.to(device)
#             return input_ids
#         preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

#     logger.info("Run calibration")
#     for i in tqdm(range(num_samples)):
#         input_ids = preprocess_fn(calibration_dataset[i])
#         # prepare inference cache for getting ssd_state scales
#         prompt_len = input_ids.size(1)
#         inf_cache = model.allocate_inference_cache(1, prompt_len)
#         lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
#         inference_params = InferenceParams(
#             max_seqlen=prompt_len,
#             max_batch_size=1,
#             seqlen_offset=0,
#             key_value_memory_dict=inf_cache,
#             lengths_per_sample=lengths_per_sample,
#         )
#         # do not set num_last_tokens because we want all activations to lm_head
#         model(input_ids, inference_params=inference_params)
#         # clean up the cache
#         del inf_cache
    
#     for h in hooks:
#         h.remove()
    
#     ##DO
#     fp_means = {}
#     for k,(collector,handle) in fp_proj_mean_hooks.items():
#         handle.remove()
#         fp_means[k] = collector.mean
    
#     # collect in/output scaling factors for layers, num_layer + lm_head
#     act_scales = [{} for _ in range(len(layers) + 1)]
#     for i in range(len(layers) + 1):
#         for name, observer in observers[i].items():
#             scale, base = observer.get_quantization_parameters()
#             # FIXME (HY): hardcode to not use base now
#             act_scales[i][name] = scale.to(torch.float32)
#         if i < len(layers):
#             col = getattr(get_mamba(layers[i]), "_collect_obs", None)
#             if col is not None:
#                 act_scales[i]["chunk_state:x_scale_chp"] = col["x_chp"].get_scale()  # (C,H,P)
#                 act_scales[i]["chunk_state:ori_x_scale_chp"] = col["ori_x_chp"].get_scale()  # (C,H,P)
#                 act_scales[i]["chunk_state:B_scale_cgn"] = col["B_cgn"].get_scale()  # (C,G,N)
#                 act_scales[i]["chunk_scan:C_scale"] = col["C_cali"].get_scale()  # (C,H,P)
#                 act_scales[i]["chunk_scan:cb_scale"] = col["cb_cali"].get_scale()  # (C,H,P)
#                 act_scales[i]["ssd_combined:state_scale"] = col["state_cali"].get_scale()  # (C,H,P)
#                 # 메모리/부작용 방지
#                 delattr(get_mamba(layers[i]), "_collect_obs")
#     del observers
#     return act_scales, fp_means ##DO

# @torch.no_grad()
# def run_quamba2_calibration(
#         model, model_type, tokenizer, reorder_params,
#         num_samples=512, seq_len=512, calibration_dataset=None, preprocess_fn=None
#     ):

#     if model_type == "mamba":
#         raise NotImplementedError("Not support for mamba")
#     elif model_type == "mamba2":
#         layers = model.backbone.layers
#         is_traget_block = lambda block: isinstance(block, Block)
#         get_mamba = lambda block: block.mixer
#         is_x = lambda op: op == "x_conv_out"
#         is_BC = lambda op: op == "B_conv_out" or op == "C_conv_out"
#         is_ssm_state = lambda op: op == "ssm_state_act"
#         is_calib_ops = lambda op: isinstance(op, (torch.nn.Linear, ActIdentity))
#         percentile_alpha=0.99999
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2'")

#     # register min/max observers, num_layer + lm_head
#     observers = [{} for _ in range(len(layers) + 1)]
    
#     def stat_hook(m, inputs, outputs, op, block_idx):
#         # register the new information to observer
#         if isinstance(inputs, tuple):
#             inputs = inputs[0]
#         observers[block_idx][op + ":input"].update(inputs.clone().detach())

#         if isinstance(outputs, tuple):
#             outputs = outputs[0]
#         observers[block_idx][op + ":output"].update(outputs.clone().detach())

#     hooks = []
#     for i in range(len(layers)):
#         if not is_traget_block(layers[i]):
#             continue
#         head_groups = reorder_params["head_groups"][i]
#         channel_group = reorder_params["channel_group"][i]
#         mixer = get_mamba(layers[i])
#         for name, m in mixer.named_modules():
#             if is_calib_ops(m):
#                 # FIXME(HY): hardcode everything for now
#                 a_bits = 8
#                 a_clip_ratio = 1.0
#                 op = name.split(".")[0]
#                 if is_x(op):
#                     observers[i][op + ":input"] = CrossHeadMinmaxObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True,
#                         ngroups=mixer.ngroups,
#                         headdim=mixer.headdim,
#                         head_groups=head_groups,
#                         channel_group=channel_group,
#                     )
#                 elif is_BC(op):
#                     observers[i][op + ":input"] = PerSSDGroupObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True,
#                         dstate=mixer.d_state,
#                     )
#                 elif is_ssm_state(op):
#                     observers[i][op + ":input"] = CachedStatesCrossHeadMinmaxObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True,
#                         ngroups=mixer.ngroups,
#                         headdim=mixer.headdim,
#                         dstate=mixer.d_state,
#                         head_groups=head_groups,
#                         channel_group=channel_group,
#                     )
#                 else:
#                     observers[i][op + ":input"] = PerTensorMinmaxObserver(
#                         n_bits=a_bits,
#                         clip_ratio=a_clip_ratio,
#                         sym=True
#                     )
#                 observers[i][op + ":output"] = PerTensorMinmaxObserver(
#                     n_bits=a_bits,
#                     clip_ratio=a_clip_ratio,
#                     sym=True
#                 )
#                 hooks.append(
#                     m.register_forward_hook(partial(stat_hook, op=op, block_idx=i))
#                 )
#     # add observer hook for lm_head
#     observers[-1]["lm_head:input"] = PerTensorMinmaxObserver(
#         n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
#     observers[-1]["lm_head:output"] = PerTensorMinmaxObserver(
#         n_bits=a_bits, clip_ratio=a_clip_ratio, sym=True)
#     hooks.append(
#         model.lm_head.register_forward_hook(partial(stat_hook, op="lm_head", block_idx=-1))
#     )
#     if calibration_dataset is None:
#         logger.info("Calibrate with monology/pile-uncopyrighted")
#         calibration_dataset = load_dataset("monology/pile-uncopyrighted", data_files="val.jsonl.zst", split="train")
#         calibration_dataset.shuffle(seed=42)

#         device = next(model.parameters()).device
#         def preprocess(data, tokenizer, max_tokens, device):
#             input_ids = tokenizer(data["text"], return_tensors="pt",
#                     max_length=max_tokens, truncation=True).input_ids.to(device)
#             return input_ids
#         preprocess_fn = partial(preprocess, tokenizer=tokenizer, max_tokens=seq_len, device=device)

#     logger.info("Run calibration")
#     for i in tqdm(range(num_samples)):
#         input_ids = preprocess_fn(calibration_dataset[i])
#         # prepare inference cache for getting ssm_state scales
#         prompt_len = input_ids.size(1)
#         inf_cache = model.allocate_inference_cache(1, prompt_len)
#         lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
#         inference_params = InferenceParams(
#             max_seqlen=prompt_len,
#             max_batch_size=1,
#             seqlen_offset=0,
#             key_value_memory_dict=inf_cache,
#             lengths_per_sample=lengths_per_sample,
#         )
#         # do not set num_last_tokens because we want all activations to lm_head
#         model(input_ids, inference_params=inference_params)
#         # clean up the cache
#         del inf_cache
    
#     for h in hooks:
#         h.remove()
    
#     # collect in/output scaling factors for layers, num_layer + lm_head
#     act_scales = [{} for _ in range(len(layers) + 1)]
#     for i in range(len(layers) + 1):
#         for name, observer in observers[i].items():
#             scale, base = observer.get_quantization_parameters()
#             # FIXME (HY): hardcode to not use base now
#             act_scales[i][name] = scale

#     return act_scales

# @torch.no_grad()
# def fuse_had_matrices(model):
#     # fuse the had matrices with the weight matrices in linear layers.
#     # Do this after reordering and before applying gptq
#     layers = model.backbone.layers
#     for i in range(len(layers)):
#         # in_proj: fuse had matrices with weight matrices
#         if isinstance(layers[i].mixer.in_proj, HadLinear):
#             layers[i].mixer.in_proj.fuse_hadamard()
#         # out_proj: fuse had matrices with weight matrices
#         if isinstance(layers[i].mixer.out_proj, HadLinear):
#             layers[i].mixer.out_proj.fuse_hadamard()
#     return model

# @torch.no_grad()
# def apply_gptq(model, tokenizer, device,args,w_bits=4,):
#     # Hardcode gptq hyper-parameters for now
#     nsamples = 128
#     seqlen = 1024
#     bits = w_bits
#     assert bits in [4, 8], "Only support 4 or 8 bits weights for now"
#     logging.info("Start Quantized Linear Layers with GPTQ")
#     logging.info("* Number of samples: %d" % nsamples)
#     logging.info("* Sequence length: %d" % seqlen)
#     logging.info("* Target bit-width for weights: %d" % bits)
#     logging.info("Build calibration loader for GPTQ")
#     #build dataloader
#     dataloader, _ = get_loaders("wikitext2", tokenizer, nsamples=nsamples, seqlen=seqlen)
#     layers = model.backbone.layers
#     model.backbone.embedding = model.backbone.embedding.to(device)
#     layers[0] = layers[0].to(device)
#     dtype = next(iter(model.parameters())).dtype
#     inps = torch.zeros(
#         (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
#     )    
#     residual = torch.zeros(
#         (nsamples, seqlen, model.config.d_model), dtype=dtype, device=device
#     )    

#     cache = {"i": 0}
#     class Catcher(nn.Module):
#         def __init__(self, module):
#             super().__init__()
#             self.module = module  
#         def forward(self, inp, res = None, **kwargs):
#             inps[cache['i']] = inp
#             cache['i'] += 1
#             raise ValueError
        
#     layers[0] = Catcher(layers[0])
#     for batch in dataloader:
#         try:
#             model(batch[0].to(device))
#         except ValueError:
#             pass

#     # the hook to collect inputs for in_proj, out_proj, and lm_head
#     def add_batch(module, inp, out, gptq):
#         gptq.add_batch(inp[0].data, out.data)

#     layers[0] = layers[0].module # remove Catcher
#     layers[0] = layers[0].cpu()
#     model.backbone.embedding = model.backbone.embedding.cpu()
#     torch.cuda.empty_cache()
#     for i in tqdm(range(len(layers))):
#         # get layer
#         layer = layers[i].to(device)
#         #print(f"BeforeQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight.norm())
#         # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
#         #print(f"BeforeQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight)
#         # torch.set_printoptions(profile='default')
#         if args and args.lowrank:
#             W = layer.mixer.out_proj.weight.data.clone()                                        #W동일
#             branch  = LowRankBranch(W.shape[1], W.shape[0],
#                                     rank=args.lr_rank, weight=W)
#             layer.mixer.lowrank_branch = branch             # 훅용
#             # layer.mixer.SVDweight = (W - branch.get_effective_weight())  # ← R
#             #print("In GPTQ, W",layer.mixer.out_proj.weight)
#             layer.mixer.out_proj.weight.data.copy_(W - branch.get_effective_weight())  # ← R
#             #print("In GPTQ, R",layer.mixer.out_proj.weight)
#             del W
#         # create GPTQ objects for in_proj and out_proj
#         gptq = {
#             "in_proj": GPTQ(layer.mixer.in_proj),
#             "out_proj": GPTQ(layer.mixer.out_proj),
#         }
#         handles = [
#             layer.mixer.in_proj.register_forward_hook(partial(add_batch, gptq=gptq["in_proj"])),
#             layer.mixer.out_proj.register_forward_hook(partial(add_batch, gptq=gptq["out_proj"]))
#         ]


#         for j in range(nsamples):
#             layer(
#                 inps[j].unsqueeze(0), 
#                 residual=residual[j].unsqueeze(0)
#             )
#             # print("residual=",residual[j].unsqueeze(0))
#         for h in handles:
#             h.remove()   
                                          
        
#         # start running GPTQ
#         for name in gptq.keys():
#             logging.debug(f"Performing GPTQ on layer.{i}.mixer.{name} with {bits} bits")
#             gptq[name].fasterquant(
#                 percdamp=0.01, group_size=128, w_bits=bits
#             )
#             gptq[name].free()
#         del gptq
        
#         # collect the outputs for the next layer
#         for j in range(nsamples):
#             inps[j], residual[j] = layer(inps[j].unsqueeze(0), residual=residual[j].unsqueeze(0))
        
#         # garbage collection and clean cache
#         layers[i] = layer.cpu()
#         #print(f"AFter Quantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight.norm())
#         # torch.set_printoptions(threshold=float('inf'))  # 모든 원소 출력
#         #print(f"AfterQuantizing layer {i} with {layer.__class__.__name__}",layer.mixer.out_proj.weight)
#         # torch.set_printoptions(profile='default')
#         del layer
#         torch.cuda.empty_cache()
#         gc.collect()

#     model = model.to("cpu") # move model to cpu to save memory
#     model.lm_head = model.lm_head.to(device)
#     model.backbone.norm_f = model.backbone.norm_f.to(device)
#     logging.info("Quantizing lm_head with GPTQ")
#     gptq_lm_head = GPTQ(model.lm_head)
#     handle = model.lm_head.register_forward_hook(partial(add_batch, gptq=gptq_lm_head))
    
#     assert model.backbone.fused_add_norm, "Only support fused_add_norm=True for now"
#     #Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L202
#     final_hidden_states = layer_norm_fn(
#         x=inps,
#         weight=model.backbone.norm_f.weight,
#         bias=model.backbone.norm_f.bias,
#         eps=model.backbone.norm_f.eps,
#         residual=residual,
#         prenorm=False,
#         residual_in_fp32=model.backbone.residual_in_fp32,
#         is_rms_norm=isinstance(model.backbone.norm_f, RMSNorm),
#     )
    
#     for j in range(nsamples):
#         model.lm_head(final_hidden_states[j].unsqueeze(0))

#     handle.remove()
#     # compute with fp16 to save memory
#     gptq_lm_head.fasterquant(
#         percdamp=0.01, group_size=128, dtype=torch.float16
#     )
#     gptq_lm_head.free()
#     del gptq_lm_head
    
#     torch.cuda.empty_cache()
#     gc.collect()

#     model = model.to(device)
#     return model


# def quantize_norm_a8(block_type, norm, layer_idx, act_scales, device):
#     norm = QRMSNorm.from_fp16(
#         originalLayer=norm,
#         output_scale=act_scales[layer_idx]["in_proj:input"].item())
#     return norm.to(device)

# def quantize_mixer_w8a8(block_type, mixer, layer_idx, act_scales,inp_smoothing,out_smoothing,use_had_transform, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
#     W8A8Mixers = {
#         "Mamba": W8A8QMamba,
#         "Mamba2": W8A8QMamba2,
#     }
#     if block_type not in W8A8Mixers.keys():
#         raise ValueError(f"Not find {block_type} in W8A8 Mixer")
#     if W8A8Mixers[block_type] is None:
#         raise ValueError(f"Not support {block_type} with W8A8")
#     mixer = W8A8Mixers[block_type].from_fp16(
#                 originalLayer=mixer,
#                 act_scales=act_scales[layer_idx],
#                 # use_had_transform=True,
#                 use_lowrank=use_lowrank,
#                 use_had_transform=use_had_transform,  
#                 inp_smoothing=inp_smoothing,
#                 lr_rank=lr_rank,
#                 out_smoothing=out_smoothing,
                
#                 use_mixed_pre=use_mixed_pre,
#                 use_squeeze=use_squeeze)
#     return mixer.to(device)

# def quantize_mixer_w4a16(block_type, mixer, layer_idx, act_scales, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
#     W4A16Mixers = {
#         "Mamba": W4A16QMamba,
#         "Mamba2": W4A16QMamba2,
#     }
#     if block_type not in W4A16Mixers.keys():
#         raise ValueError(f"Not find {block_type} in W4A16 Mixer")
#     if W4A16Mixers[block_type] is None:
#         raise ValueError(f"Not support {block_type} with W4A16")
#     mixer = W4A16Mixers[block_type].from_fp16(
#                 originalLayer=mixer, 
#                 use_had_transform=True,
#                 use_lowrank=use_lowrank,   
#                 lr_rank=lr_rank,
#                 use_mixed_pre=use_mixed_pre,
#                 use_squeeze=use_squeeze)
#     return mixer.to(device)

# def quantize_mixer_w4a8(block_type, mixer, layer_idx, act_scales,out_smoothing,inp_smoothing,use_had_transform, device,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False):
#     W4A8Mixers = {
#         "Mamba": W4A8QMamba,
#         "Mamba2": W4A8QMamba2,
#     }
#     if block_type not in W4A8Mixers.keys():
#         raise ValueError(f"Not find {block_type} in W4A8 Mixer")
#     if W4A8Mixers[block_type] is None:
#         raise ValueError(f"Not support {block_type} with W4A8")
#     #print("[DEBUG] quantize_mixer_w4a8 use_lowrank =", use_lowrank)
#     mixer = W4A8Mixers[block_type].from_fp16(
#                 originalLayer=mixer,
#                 act_scales=act_scales[layer_idx],
#                 # use_had_transform=True,
#                 use_lowrank=use_lowrank,   
#                 lr_rank=lr_rank,
#                 use_mixed_pre=use_mixed_pre,
#                 out_smoothing=out_smoothing,
#                 inp_smoothing=inp_smoothing,
#                 use_had_transform=use_had_transform,
#                 use_squeeze=use_squeeze)
#     return mixer.to(device)

# def get_quantize_block_fn(act_scales, w_bits, a_bits, device,inp_smoothing, out_smoothing, use_had_transform):
#     if w_bits == 4 and a_bits == 8:
#         quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
#         quantize_mixer_fn = partial(quantize_mixer_w4a8, act_scales=act_scales, device=device,inp_smoothing=inp_smoothing, out_smoothing=out_smoothing, use_had_transform=use_had_transform)
#     elif w_bits == 4 and a_bits == 16:
#         quantize_norm_fn = lambda block_type, norm, layer_idx: norm # just return the original layer
#         quantize_mixer_fn = partial(quantize_mixer_w4a16, act_scales=act_scales, device=device)
#     elif w_bits == 8 and a_bits == 8:
#         quantize_norm_fn = partial(quantize_norm_a8, act_scales=act_scales, device=device)
#         quantize_mixer_fn = partial(quantize_mixer_w8a8, act_scales=act_scales, device=device,inp_smoothing=inp_smoothing, out_smoothing=out_smoothing, use_had_transform=use_had_transform)
#     else:
#         raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
#     return quantize_norm_fn, quantize_mixer_fn
    
# @torch.no_grad()
# def quantize_fp16_model(model, model_type, act_scales, device, w_bits=4, a_bits=8, quantize_embedding=True, quantize_lm_head=True,use_lowrank=False, lr_rank=32,use_mixed_pre=False,use_squeeze=False, inp_smoothing=False,out_smoothing=False, use_had_transform=False):
#     assert w_bits in [1, 4, 8], "Only support 4 or 8 bits weights for now"
#     assert a_bits in [1, 8, 16], "Only support 8 or 16 bits activations for now"
#     quantize_norm_fn, quantize_mixer_fn = get_quantize_block_fn(act_scales, w_bits, a_bits, device, inp_smoothing, out_smoothing, use_had_transform)
#     model.config.use_cache = False
#     if model_type == "mamba":
#         if quantize_embedding:
#             # replace embedding layer
#             logging.info(f'Applying quantized embedding')
#             if w_bits == 4:
#                 model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
#             elif w_bits == 8:
#                 model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
#             else:
#                 raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
#             gc.collect()
#             torch.cuda.empty_cache()
#         # replace layers
#         logging.info(f'Applying quantized layers')
#         layers = model.backbone.layers
#         for i in tqdm(range(len(layers))):
#             if isinstance(layers[i], Block):
#                 # replace with fused RMSNorm
#                 #print(f"[DEBUG] layer {i} use_lowrank in quantize_fp16_model:", use_lowrank)
#                 layers[i].fused_add_norm = True
#                 layers[i].norm = quantize_norm_fn(
#                     block_type="Mamba",
#                     norm=layers[i].norm,
#                     layer_idx=i)
#                 layers[i].mixer = quantize_mixer_fn(
#                     block_type="Mamba", 
#                     mixer=layers[i].mixer,
#                     layer_idx=i,
                    
#                     )
#                 # garbage collection and clean cache
#                 gc.collect()
#                 torch.cuda.empty_cache()
#     elif model_type == "mamba2":
#         if quantize_embedding:
#             # replace embedding layer
#             logging.info(f'Applying quantized embedding')
#             if w_bits == 4:
#                 model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
#             elif w_bits == 8:
#                 model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
#             elif w_bits == 1:                                                                       ##일단 88취급
#                 model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
                
#             else:
#                 raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
#             gc.collect()
#             torch.cuda.empty_cache()
#         # replace layers
#         logging.info(f'Applying quantized layers')
#         layers = model.backbone.layers
#         for i in tqdm(range(len(layers))):
#             if isinstance(layers[i], Block):
#                 layers[i].fused_add_norm = True
#                 if not inp_smoothing: ##DO## for debugging
#                     layers[i].norm = quantize_norm_fn(
#                             block_type="Mamba2",
#                             norm=layers[i].norm,
#                             layer_idx=i)
#                 layers[i].mixer = quantize_mixer_fn(
#                     block_type="Mamba2", 
#                     mixer=layers[i].mixer,
#                     layer_idx=i,
#                     use_lowrank=use_lowrank,   # 새 인자
#                     lr_rank=lr_rank,
#                     use_mixed_pre=use_mixed_pre,
#                     use_squeeze=use_squeeze)
#                 # garbage collection and clean cache
#                 gc.collect()
#                 torch.cuda.empty_cache()
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")
    
#     if quantize_lm_head:
#         logging.info(f'Applying quantized lm_head')
#         # replace lm_head and norm_f with quantized version
#         if w_bits == 4 and a_bits == 16:
#             # do nothing to w4a16 norm_f
#             model.lm_head = W4A16B16O16Linear.from_fp16(model.lm_head)
#         elif w_bits == 4 and a_bits == 8:
#             # model.backbone.norm_f = FusedRMSNorm.from_fp16(
#             model.backbone.norm_f = QRMSNorm.from_fp16(
#                 model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
#             model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
#         elif w_bits == 8 and a_bits == 8:
#             # model.backbone.norm_f = FusedRMSNorm.from_fp16(
#             model.backbone.norm_f = QRMSNorm.from_fp16(
#                 model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
#             model.lm_head = W8A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"].item())
#         elif w_bits == 1 and a_bits == 1:                                                                   #일단 88 취급
#             # model.backbone.norm_f = FusedRMSNorm.from_fp16(
#             model.backbone.norm_f = QRMSNorm.from_fp16(
#                 model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
#             model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
#         else:
#             raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
    
#     gc.collect()
#     torch.cuda.empty_cache()
#     return model

# def quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=4, 
#                                    layer_wise_hybrid_config=None #this is expected to be a list
#                                    ):
#     assert w_bits in [4], "Only support 4 bits weights for now"
#     a_bits = [8, 16]
    
#     logging.info(f"Quantizing model with w{w_bits} and a{a_bits}. HybridBlock will be create.")
    
#     # for each a_bits get the correcponding quantization function
#     quant_function_pairs = {}
#     for a in a_bits:
#         quant_function_pairs[f'W{w_bits}A{a}'] = get_quantize_block_fn(act_scales, w_bits, a, device) # quantize_norm_fn, quantize_mixer_fn
    
#     model.config.use_cache = False
#     if model_type == "mamba2":
#         # replace embedding layer
#         logging.info(f'Applying quantized embedding')
#         if w_bits == 4:
#             model.backbone.embedding = W4O16Embedding.from_fp16(model.backbone.embedding)
#         elif w_bits == 8:
#             model.backbone.embedding = W8O16Embedding.from_fp16(model.backbone.embedding)
#         else:
#             raise ValueError(f"Unsupport w{w_bits}a{a_bits}, only w8a8, w4a8, and w4a16 are supported")
#         # replace layers
#         logging.info(f'Applying quantized layers')
#         layers = model.backbone.layers
#         for i in tqdm(range(len(layers))):
#             if isinstance(layers[i], Block):
#                 layers[i].fused_add_norm = True
                
#                 mixers = {}
#                 norms = {}
                
#                 if layer_wise_hybrid_config:
#                     #Case 1: bitwidth of each layer is specified, each layers only create the block/norm with the specified bitwidth
#                     bit_config = layer_wise_hybrid_config[i]
#                     (quantize_norm_fn, quantize_mixer_fn) = quant_function_pairs[bit_config]
#                     mixers[bit_config] = quantize_mixer_fn(
#                         block_type="Mamba2", 
#                         mixer=layers[i].mixer,
#                         layer_idx=i)
#                     norms[bit_config] = quantize_norm_fn(
#                         block_type="Mamba2",
#                         norm=layers[i].norm,
#                         layer_idx=i)     
#                 else:
#                     #Case 2: bitwidth of each layer is not specified, each layers create the block/norm with all possible bitwidth
#                     for bits, (quantize_norm_fn, quantize_mixer_fn) in quant_function_pairs.items():
#                         mixers[f"W{w_bits}A{bits}"] = quantize_mixer_fn(
#                             block_type="Mamba2", 
#                             mixer=layers[i].mixer,
#                             layer_idx=i)
#                         norms[f"W{w_bits}A{bits}"] = quantize_norm_fn(
#                             block_type="Mamba2",
#                             norm=layers[i].norm,
#                             layer_idx=i)          
                            
#                 layers[i] = HybridQuambaBlock.from_block_and_mixer_norm_dict(
#                     block=layers[i],
#                     mixers_dict=mixers,
#                     norms_dict=norms,
#                 ).to(device)
#                 layers[i].set_mixer(next(iter(mixers))) #default
#                 gc.collect()
#                 torch.cuda.empty_cache()
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba2' for hybrid mode")
    
#     logging.info(f'Applying quantized lm_head')
    
#     #FIXME(brian1009): Hard-fix for now, but we may need to make it configurable as well.
#     model.backbone.norm_f = QRMSNorm.from_fp16(
#         model.backbone.norm_f, output_scale=act_scales[-1]["lm_head:input"].item())
#     model.lm_head = W4A8B16O16Linear.from_fp16(model.lm_head, act_scales[-1]["lm_head:input"])
    
#     gc.collect()
#     torch.cuda.empty_cache()
#     return model


# def get_model_size(model, model_name, w_bits, a_bits):
#     param_size = 0
#     for param in model.parameters():
#         param_size += param.nelement() * param.element_size()
#     buffer_size = 0
#     for buffer in model.buffers():
#         buffer_size += buffer.nelement() * buffer.element_size()
#     model_mb = (param_size + buffer_size) / 1024**2
#     logging.info(f'W{w_bits}A{a_bits} {model_name} size: {model_mb:.3f} MB')


# def quantize_model_mamba(model, model_type, tokenizer, device, args, calibration_dataset=None, calib_preprocess_fn=None,):
#     # restore the quantized model from pretrained_dir
#     if args.pretrained_dir:
#         # change the name to lookup the model in the pretrained_dir
#         model_name = args.model.lower().split('/')[-1]

#         # already loaded the quantized model in build_mamba_and_tokenizer in utils.py
#         if model_name.startswith("quamba"): # ut-enyac/quamba or ut-enyac/quamba2
#             get_model_size(model, args.model, args.w_bits, args.a_bits)
#             return model

#         # load quantied model in args.pretrained_dir to replace fp16 mamba
#         # This will require much more memory, since we will have
#         # fp16 mamba, qumaba, and quamba weight in the memory at the same time
#         if model_name.startswith("mamba"): # mamba or mamba2
#             model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
#             if args.hybrid_blocks: 
#                 model_name = model_name + f"-w{args.w_bits}aX"
#                 if args.hybrid_blocks_config:
#                     config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
#                     model_name = model_name + f"-{config_name}"
#             else:
#                 model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"
#             quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
#         else:
#             logging.warning(f"Unsupported model {args.model} in ut-enyac/ model registry")
#         # load the quantized model if it exists
#         if os.path.isdir(quantized_model_path):
#             logging.info(f"Loading quantized model from {quantized_model_path}")
#             model = QuambaLMHeadModel.from_pretrained(quantized_model_path, device="cuda")
#             get_model_size(model, args.model, args.w_bits, args.a_bits)
#             return model
#         else:
#             logging.warning(f"{quantized_model_path} does not exist.")
#             logging.warning("Runing calibration and quantization from scratch")
#     if args.apply_smoothing:
#         args.apply_inp_smoothing, args.apply_out_smoothing = True, True
#         args.smoothing_inp_alpha, args.smoothing_out_alpha = args.smoothing_alpha, args.smoothing_alpha
#     # replace the mamba blocks with simple blocks to get the scaling factors
#     # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
#     model = configure_model(model, model_type, use_had_transform=args.apply_hadamard, inp_smoothing=args.apply_inp_smoothing,out_smoothing=args.apply_out_smoothing)   ##DO## # W4A16 needs had_transform as well #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#     """아다마르 변환 일단 적용 X"""
#     "smoothing point" ##DO##
#         ###########################################################가중치 시각화##################################################
#     # import torch, matplotlib.pyplot as plt, numpy as np
#     # from pathlib import Path
#     # def plot_row_stats(
#     #     W: torch.Tensor,
#     #     name="W",
#     #     save_dir="row_plots",
#     #     y_lim=None,                          # ← 기본값 None = 자동 스케일
#     #     colors=("forestgreen", "goldenrod", "maroon"),
#     #     dpi=140):

#     #     W = W.detach().float().cpu()
#     #     med  = W.median(dim=0).values
#     #     p99  = torch.quantile(W, .99, dim=0)
#     #     vmax = W.max(dim=0).values
#     #     xs   = np.arange(W.shape[1])

#     #     fig, ax = plt.subplots(figsize=(9,3), dpi=dpi)
#     #     ax.fill_between(xs, 0, med,  color=colors[0], alpha=.8, label="median")
#     #     ax.fill_between(xs, med, p99, color=colors[1], alpha=.8, label="99 %")
#     #     ax.fill_between(xs, p99, vmax,color=colors[2], alpha=.8, label="max")

#     #     ax.set(xlabel="col index",
#     #         ylabel="Weight value",
#     #         title=f"{name} — per-col stats")
#     #     if y_lim is not None:                   # ← 자동/고정 선택
#     #         ax.set_ylim(*y_lim)
#     #     ax.margins(x=0)
#     #     ax.legend(fontsize=8, framealpha=.9)
#     #     fig.tight_layout()

#     #     Path(save_dir).mkdir(exist_ok=True, parents=True)
#     #     fname = Path(save_dir) / f"{name.lower()}_row_stat_smoothba.png"
#     #     fig.savefig(fname); plt.close(fig)
#     #     print("✓ saved →", fname)
# ###############################################################################가중치시각화#############################################################
#     #W_smooth_before = model.backbone.layers[-1].mixer.out_proj.weight
#     #print("BeforeSmooth",model.backbone.layers[-1].mixer.out_proj.weight)
#     #plot_row_stats(W_smooth_before, "W_smooth_before")
#     for i, blk in enumerate(model.backbone.layers):
#         W_blk = blk.mixer.out_proj.weight
#         #print(f"Layer {i} - before had(smooth) weight norm: {W_blk.norm()}")
#     if args.apply_inp_smoothing or args.apply_out_smoothing:
#         logging.info(f"Start doing smoothing")
#         smooth_mamba(model, tokenizer, num_samples=5 if args.testing else 512, inp_smoothing = args.apply_inp_smoothing, out_smoothing=args.apply_out_smoothing,
#                      inp_alpha=args.smoothing_inp_alpha, out_alpha=args.smoothing_out_alpha)
#         #print("AfterSmooth",model.backbone.layers[-1].mixer.out_proj.weight)
#         #W_smooth_after = model.backbone.layers[-1].mixer.out_proj.weight
#         #plot_row_stats(W_smooth_after,      "W_smooth_after")
#     # replace the mamba blocks with simple blocks to get the scaling factors
#     # we hardcode use_had_transform=True to fix the configuration, so it is easier for users
#     ##############################################original code#############################################################
#     #model = configure_model(model, model_type, use_had_transform=True) # W4A16 needs had_transform as well
#     #####################################################################################################
#     logging.info(f"Target bit-width W{args.w_bits}A{args.a_bits}")
#     if args.a_bits == 8:
#         # Run calibration to get scale and reorder params
#         if args.group_heads:
#             logging.info(f"Reordering weights and activations for head grouping")
#             reorder_params = get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512)
#             reorder_mamba(model, reorder_params)
#             # collect 8-bit activation scales
#             act_scales = run_quamba2_calibration(model, model_type, tokenizer, reorder_params,
#                                                 num_samples=args.calib_data_num,
#                                                 seq_len=args.calib_seqlen,
#                                                 calibration_dataset=calibration_dataset,
#                                                 preprocess_fn=calib_preprocess_fn)
#         else:
#             # collect 8-bit activation scales
#             act_scales, fp_means = run_quamba_calibration(model, model_type, tokenizer,
#                                                 num_samples=args.calib_data_num,
#                                                 seq_len=args.calib_seqlen,
#                                                 calibration_dataset=calibration_dataset,
#                                                 preprocess_fn=calib_preprocess_fn)
#             # print("fp_means:",fp_means)
#     elif args.a_bits == 16:
#         # not doing anything for activations
#         act_scales = {}
#         if args.compensation :
#             logging.info(f"Collecting fp means for compensation")
#             _, fp_means = run_quamba_calibration(model, model_type, tokenizer,
#                                                 num_samples=args.calib_data_num,
#                                                 seq_len=args.calib_seqlen,
#                                                 calibration_dataset=calibration_dataset,
#                                                 preprocess_fn=calib_preprocess_fn)
#             # print("fp_means:",fp_means)
#         if args.group_heads:
#             logging.info(f"Activation bit-width is set to 16, skip weights reordering and head grouping")
#     elif args.a_bits == 1:
#         # Run calibration to get scale and reorder params
#         if args.group_heads:
#             logging.info(f"Reordering weights and activations for head grouping")
#             reorder_params = get_reorder_params(model, model_type, tokenizer, num_samples=512, seq_len=512)
#             reorder_mamba(model, reorder_params)
#             # collect 8-bit activation scales
#             act_scales = run_quamba2_calibration(model, model_type, tokenizer, reorder_params,
#                                                 num_samples=args.calib_data_num,
#                                                 seq_len=args.calib_seqlen,
#                                                 calibration_dataset=calibration_dataset,
#                                                 preprocess_fn=calib_preprocess_fn)
#         else:
#             # collect 8-bit activation scales
#             act_scales = run_quamba_calibration(model, model_type, tokenizer,
#                                                 num_samples=args.calib_data_num,
#                                                 seq_len=args.calib_seqlen,
#                                                 calibration_dataset=calibration_dataset,
#                                                 preprocess_fn=calib_preprocess_fn)
#     else:
#         raise ValueError(f"Unsupported activation bit-width: {args.a_bits}, try --a_bits 8 or --a_bits 16?")
    

#     # fuse the had matrices with the weight matrices in linear layers.
#     # Do this after reordering and before applying gpt
#     model = fuse_had_matrices(model) 
#     #print("AfterSMOOTH FUSE ",model.backbone.layers[-1].mixer.out_proj.weight)
#     # if args.lowrank and not args.apply_gptq:
#     #     for blk in model.backbone.layers:
#     #         lr_rank = args.lr_rank

#     #         # ── 1. GPU weight → W_fp16
#     #         W_fp16 = blk.mixer.out_proj.weight.data          # (GPU - fp16)

#     #         # ── 2. GPU에서 branch 만들고 잔차 R 계산
#     #         branch = LowRankBranch(
#     #             in_features=W_fp16.shape[1],
#     #             out_features=W_fp16.shape[0],
#     #             rank=lr_rank,
#     #             weight=W_fp16,                               # <- GPU tensor
#     #         )

#     #         R = W_fp16 - branch.get_effective_weight()       # R ⟵ GPU 계산
#     #         blk.mixer.out_proj.weight.data.copy_(R)          # 잔차로 덮어쓰기

#     #         # ── 3. branch 는 CPU 로 옮겨서 저장, GPU 메모리 해제
#     #         blk.mixer.lowrank_branch = branch.cpu()        # branch(fp16, CPU)
#     #         del R                                            # 덜 필요하지만 확실히
#     #         torch.cuda.empty_cache()
#     # Apply GPTQ to quantize linear
#     #print("BeforeGPTQ",model.backbone.layers[-1].mixer.out_proj.weight)
#     if args.apply_gptq:
#         model = apply_gptq(model, tokenizer, device, args , w_bits=args.w_bits)
    
    
#     # Replace (reordered, fused, and GPTQ quantized) modules with quantized version
#     #print("AfterGPTQ",model.backbone.layers[-1].mixer.out_proj.weight)
#     if args.hybrid_blocks: # create hybrid block
#         if args.hybrid_blocks_config:
#             logging.info(f"Loading hybrid blocks config from {args.hybrid_blocks_config}")
#             with open(args.hybrid_blocks_config, "r") as file:
#                 hybrid_blocks_configs = json.load(file)
#         else:
#             hybrid_blocks_configs = None
#         model = quantize_fp16_model_act_hybrid(model, model_type, act_scales, device, w_bits=args.w_bits,
#                                                layer_wise_hybrid_config=hybrid_blocks_configs)
#     else:
#         # print("[DEBUG] args.lowrank =", args.lowrank)  # <-- 여기
#         model = quantize_fp16_model(
#             model, model_type, act_scales, device,
#             w_bits=args.w_bits, a_bits=args.a_bits,
#             quantize_embedding=args.quantize_embedding,
#             quantize_lm_head=args.quantize_lm_head,
#             use_lowrank=args.lowrank, 
#             lr_rank=args.lr_rank,
#             use_mixed_pre=args.mixed_pre,
#             use_squeeze=args.squeeze,
#             inp_smoothing=args.apply_inp_smoothing,
#             out_smoothing=args.apply_out_smoothing,
#             use_had_transform=args.apply_hadamard
#         )
#         print("quantization complete")
        
#         #DO
#         if args.compensation :
#             model._fp_means = fp_means
#             model = mamba_sequential_compensation_ssdout(
#                 model=model,
#                 device=device,
#                 nsamples=args.comp_sam_num,
#                 tokenizer=tokenizer,
#                 seq_len=args.calib_seqlen,
#                 dataloader=None,                      # 옵션 A를 쓰면 여기에 로더
#                 dataset=calibration_dataset,          # ← run_quamba_calibration에 쓴 것과 동일
#                 preprocess_fn=calib_preprocess_fn,     # ← run_quamba_calibration에 쓴 것과 동일
#                 comp_out_decay=args.comp_out_decay,
#                 comp_ssd_decay=args.comp_ssd_decay,
#                 layers=args.comp_layers
#             )
#             model = model.to(device).eval()
#             torch.cuda.empty_cache()
        
#         # quant_means = test_quant_calibration(model, model_type, tokenizer,
#         #                                         # num_samples=args.calib_data_num,
#         #                                         num_samples = 512,
#         #                                         seq_len=args.calib_seqlen,
#         #                                         calibration_dataset=calibration_dataset,
#         #                                         preprocess_fn=calib_preprocess_fn, device=device)
#         # print("quant_means:",quant_means)
#         # return
        
#     # store the state_dict if not quamba
#     model_name = args.model.lower().split('/')[-1]
#     if args.pretrained_dir is not None and not model_name.startswith("quamba"):
#         if not args.hybrid_blocks:
#             # change the name to store the model in the pretrained_dir
#             model_name = args.model.lower().split('/')[-1]
#             model_name = model_name.replace("mamba", "quamba") # replace mamba with quamba
#             model_name = model_name + f"-w{args.w_bits}a{args.a_bits}"

#             quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
#             # we slightly hack the api: we use MambaLMHeadModel instead of QuambaLMHeadModel to store the model here
#             model.config.ssm_cfg['layer'] = model.backbone.layers[0].mixer.__class__.__name__
#             model.config.norm_cfg = {"norm": model.backbone.layers[0].norm.__class__.__name__}
#             model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
#             model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
#             # We apply Hadamard transforms so we cannot tie embeddings and lm_head
#             model.config.tie_embeddings = False # no used in QuambaLMHeadModel
#             if hasattr(model.config, "use_cache"):
#                 delattr(model.config, "use_cache")
#             model.save_pretrained(quantized_model_path)
#             logging.info(f"The quantized model is stored at {quantized_model_path}")
#         else:
#             model_name = args.model.lower().split('/')[-1]
#             model_name = model_name.replace("mamba", "quamba")
#             model_name = model_name + f"-w{args.w_bits}aX"
#             if args.hybrid_blocks_config:
#                 config_name = args.hybrid_blocks_config.split("/")[-1].replace(".json", "")
#                 model_name = model_name + f"-{config_name}"
#             quantized_model_path = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
            
#             ssm_layer_infos = []
#             norm_infos = []
            
#             for i in range(len(model.backbone.layers)):
#                 mixer_norms_info = model.backbone.layers[i].get_class_info_dict_mixers_norms()
#                 ssm_layer_infos.append(mixer_norms_info["mixers"])
#                 norm_infos.append(mixer_norms_info["norms"])
            
#             mixer_norms_info = model.backbone.layers[0].get_class_info_dict_mixers_norms()
#             model.config.ssm_cfg['layer'] = ssm_layer_infos
#             model.config.norm_cfg = {"norm": norm_infos}
#             model.config.embedding_cfg = {"layer": model.backbone.embedding.__class__.__name__}
#             model.config.lm_head_cfg = {"layer": model.lm_head.__class__.__name__}
#             # We apply Hadamard transforms so we cannot tie embeddings and lm_head
#             model.config.tie_embeddings = False # no used in QuambaLMHeadModel
#             if hasattr(model.config, "use_cache"):
#                 delattr(model.config, "use_cache")
#             model.config.is_hybrid = True
#             model.save_pretrained(quantized_model_path)
#             logging.info(f"The quantized model is stored at {quantized_model_path}")
            
#         # store tokenizer for mamba2-8b
#         if "mamba2-8b" in args.model:
#             # model.save_pretrained should already create the saved dir
#             saved_dir = os.path.join(args.pretrained_dir, "ut-enyac", model_name)
#             tokenizer.save(saved_dir)
#             logging.info(f"Tokenizer is stored at {saved_dir}")
#     # quantized model
#     get_model_size(model, args.model, args.w_bits, args.a_bits)
#     return model.to(device)







# @torch.no_grad()
# def test_quant_calibration(
#         model, model_type, tokenizer, num_samples=512, seq_len=2048,
#         calibration_dataset=None, preprocess_fn=None, device=None
#     ):
#     if model_type == "mamba":
#         layers = model.backbone.layers
#         is_traget_block = lambda block: isinstance(block, Block)
#         get_mamba = lambda block: block.mixer
#     elif model_type == "mamba2":
#         layers = model.backbone.layers
#         is_traget_block = lambda block: isinstance(block, Block)
#         get_mamba = lambda block: block.mixer
#     else:
#         raise ValueError(f"Unsupported model type: {model_type}, only support 'mamba' and 'mamba2'")

#     model.eval()
#     device = device

#     # in/out 평균 수집 훅
#     fp_proj_mean_hooks = {}

#     for i in range(len(layers)):
#         if not is_traget_block(layers[i]):
#             continue
#         mixer = get_mamba(layers[i])  # ← 필수 (주석 풀기)
#         _in_col  = _ChannelMeanCollector()
#         _out_col = _ChannelMeanCollector()
#         h_in  = mixer.in_proj.register_forward_hook(
#             lambda _m, _i, out, col=_in_col: col.update(out)
#         )
#         h_out = mixer.out_proj.register_forward_hook(
#             lambda _m, _i, out, col=_out_col: col.update(out)
#         )
#         fp_proj_mean_hooks[f"L{i}/in_proj_out"]  = (_in_col, h_in)
#         fp_proj_mean_hooks[f"L{i}/out_proj_out"] = (_out_col, h_out)  
        
#     # 데이터 준비 (기본: pile)
#     if calibration_dataset is None:
#         logger.info("Calibrate with monology/pile-uncopyrighted")
#         calibration_dataset = load_dataset(
#             "monology/pile-uncopyrighted",
#             data_files="val.jsonl.zst",
#             split="train"
#         )

#         def _preprocess(data, tokenizer, max_tokens, device):
#             return tokenizer(
#                 data["text"],
#                 return_tensors="pt",
#                 max_length=max_tokens,
#                 truncation=True
#             ).input_ids.to(device)

#         preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
#                                 max_tokens=seq_len, device=device)
#     assert preprocess_fn is not None

#     # 전방패스 수행
#     logger.info("Run calibration (test collection)")
#     for i in tqdm(range(num_samples)):
#         input_ids = preprocess_fn(calibration_dataset[i])
#         prompt_len = input_ids.size(1)
#         inf_cache = model.allocate_inference_cache(1, prompt_len)
#         lengths_per_sample = torch.full((1,), prompt_len, dtype=torch.int32, device=device)
#         inference_params = InferenceParams(
#             max_seqlen=prompt_len,
#             max_batch_size=1,
#             seqlen_offset=0,
#             key_value_memory_dict=inf_cache,
#             lengths_per_sample=lengths_per_sample,
#         )
#         model(input_ids, inference_params=inference_params)
#         del inf_cache

#     # 훅 해제 및 결과 집계
#     quant_means = {}
#     for k, (collector, handle) in fp_proj_mean_hooks.items():
#         handle.remove()
#         quant_means[k] = collector.mean

#     return quant_means





# @torch.no_grad()
# def mamba_sequential_compensation_update(
#     model,
#     device,
#     nsamples,
#     tokenizer,
#     seq_len=2048,
#     dataloader=None,
#     dataset=None,
#     preprocess_fn=None,
#     comp_in_decay=0.1,
#     comp_out_decay=0.1,
#     comp_ssd_decay=0.1,     # ← 추가: ssd 보상 감쇠
#     layers=None,            # ← 선택 레이어만 보상
# ):
#     import torch
#     from torch import nn
#     from tqdm import tqdm

#     def _batch_sum_and_count(y: torch.Tensor):
#         if y.dim() == 3: return y.sum(dim=(0,1)).float(), y.shape[0]*y.shape[1]
#         if y.dim() == 2: return y.sum(dim=0).float(), y.shape[0]
#         raise RuntimeError(f"Unexpected shape {tuple(y.shape)}")

#     model.eval()
#     blocks = model.backbone.layers
#     fp_means = getattr(model, "_fp_means", None)
#     assert isinstance(fp_means, dict)

#     # --- 캡처 준비(입력 x만) ---
#     model.backbone.embedding = model.backbone.embedding.to(device)
#     blocks[0] = blocks[0].to(device)

#     if dataloader is None and (dataset is None or preprocess_fn is None):
#         from datasets import load_dataset
#         dataset = load_dataset("monology/pile-uncopyrighted",
#                                data_files="val.jsonl.zst", split="train")
#         def _preprocess(data, tokenizer, max_tokens, device):
#             return tokenizer(data["text"], return_tensors="pt",
#                              truncation=True, max_length=max_tokens
#                             ).input_ids.to(device)
#         from functools import partial
#         preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
#                                 max_tokens=seq_len, device=device)

#     inps_list = []
#     class Catcher(nn.Module):
#         def __init__(self, mod): super().__init__(); self.mod = mod
#         def forward(self, x, *args, **kwargs):
#             inps_list.append(x.squeeze(0).detach() if (x.dim()==3 and x.size(0)==1) else x.detach())
#             raise ValueError
#     blocks[0] = Catcher(blocks[0])

#     if dataloader is not None:
#         it = iter(dataloader)
#         for _ in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             batch = next(it)
#             try:
#                 xb = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 model(xb.to(device))
#             except ValueError:
#                 pass
#     else:
#         for i in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             x = preprocess_fn(dataset[i])
#             try: model(x)
#             except ValueError: pass

#     blocks[0] = blocks[0].mod
#     blocks[0] = blocks[0].cpu()
#     model.backbone.embedding = model.backbone.embedding.cpu()
#     torch.cuda.empty_cache()
#     assert len(inps_list) == nsamples

#     # residual 초기화(모델이 residual을 반환하는 인터페이스일 경우 사용)
#     res_in_list = [None] * nsamples

#     # --- 레이어 선택 집합 ---
#     L = len(model.backbone.layers)
#     def _norm_idxs(idxs):
#         s=set()
#         for k in idxs: 
#             if k<0: k=L+k
#             if 0<=k<L: s.add(int(k))
#         return s
    
#     if layers is None:
#         target_layers = set(range(L))
#     else:
#         target_layers = _norm_idxs(list(layers) if not isinstance(layers, (list, tuple)) else layers)

#     # --- 레이어 순차 보상 ---
#     for li in tqdm(range(L), desc="[Layer] Sequential compensation", leave=True):
#         layer = blocks[li].to(device)
#         mix = layer.mixer
#         mix.compensation = True

#         # 대상이 아니면 그냥 전파
#         if li not in target_layers:
#             next_inps, next_res = [], []
#             for j in range(nsamples):
#                 xj = inps_list[j];  xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 yj, resj = layer(xj, residual=res_in_list[j])
#                 next_inps.append(yj.squeeze(0).detach())
#                 next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
#             blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#             inps_list, res_in_list = next_inps, next_res
#             continue

#         # 보상 버퍼 준비
#         if getattr(mix, "comp_in", None)  is None:
#             mix.register_buffer("comp_in",  torch.zeros(mix.in_proj.out_features,  dtype=torch.float16, device=device))
#         if getattr(mix, "ssd_comp", None) is None:
#             mix.register_buffer("ssd_comp", torch.zeros(mix.d_inner,               dtype=torch.float16, device=device))
#         if getattr(mix, "comp_out", None) is None:
#             mix.register_buffer("comp_out", torch.zeros(mix.out_proj.out_features, dtype=torch.float16, device=device))

#         # fp 기준 평균
#         m_fp_in   = fp_means.get(f"L{li}/in_proj_out",  None)
#         m_fp_ssd  = fp_means.get(f"L{li}/ssd_out",      None)  # ← 추가
#         m_fp_out  = fp_means.get(f"L{li}/out_proj_out", None)
#         if m_fp_in  is not None: m_fp_in  = m_fp_in.to(device=device, dtype=torch.float32)
#         if m_fp_ssd is not None: m_fp_ssd = m_fp_ssd.to(device=device, dtype=torch.float32)
#         if m_fp_out is not None: m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)

#         # =========================
#         # Phase A: comp_in 추정
#         # =========================
#         mix.comp_in.zero_(); mix.ssd_comp.zero_(); mix.comp_out.zero_()
#         in_sum=None; in_cnt=0
#         def in_hook(_m,_i,out):
#             nonlocal in_sum,in_cnt
#             s,n = _batch_sum_and_count(out)
#             in_sum = s if in_sum is None else (in_sum+s); in_cnt+=int(n)
#         hA = mix.in_proj.register_forward_hook(in_hook)
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase A: in_proj", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             _ = layer(xj, residual=res_in_list[j], comp_calib = True)
#         hA.remove()
#         if (m_fp_in is not None) and (in_cnt>0):
#             q_mean_in = torch.where(torch.isfinite(in_sum/in_cnt), in_sum/in_cnt, torch.zeros_like(in_sum))
#             mix.comp_in.copy_((m_fp_in - q_mean_in).to(mix.comp_in.dtype) * comp_in_decay)

#         # =========================
#         # Phase B: ssd_comp 추정
#         # (comp_in on, ssd_comp/off, comp_out/off)
#         # =========================
#         mix.ssd_comp.zero_(); mix.comp_out.zero_()
#         ssd_sum=None; ssd_cnt=0
#         if hasattr(mix, "ssd_out_act") and mix.ssd_out_act is not None and (m_fp_ssd is not None):
#             def ssd_hook(_m,_i,out):
#                 nonlocal ssd_sum, ssd_cnt
#                 s,n = _batch_sum_and_count(out)
#                 ssd_sum = s if ssd_sum is None else (ssd_sum+s); ssd_cnt+=int(n)
#             hB = mix.ssd_out_act.register_forward_hook(ssd_hook)
#             for j in tqdm(range(nsamples), desc=f"[L{li}] Phase B: ssd_out", leave=False):
#                 xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 _ = layer(xj, residual=res_in_list[j], comp_calib = True)
#             hB.remove()
#             if ssd_cnt>0:
#                 q_mean_ssd = torch.where(torch.isfinite(ssd_sum/ssd_cnt), ssd_sum/ssd_cnt, torch.zeros_like(ssd_sum))
#                 mix.ssd_comp.copy_((m_fp_ssd - q_mean_ssd).to(mix.ssd_comp.dtype) * comp_ssd_decay)

#         # =========================
#         # Phase C: comp_out 추정
#         # (comp_in+ssd_comp on, comp_out/off)
#         # =========================
#         mix.comp_out.zero_()
#         out_sum=None; out_cnt=0
#         def out_hook(_m,_i,out):
#             nonlocal out_sum,out_cnt
#             s,n = _batch_sum_and_count(out)
#             out_sum = s if out_sum is None else (out_sum+s); out_cnt+=int(n)
#         hC = mix.out_proj.register_forward_hook(out_hook)
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase C: out_proj", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             _ = layer(xj, residual=res_in_list[j], comp_calib = True)
#         hC.remove()
#         if (m_fp_out is not None) and (out_cnt>0):
#             q_mean_out = torch.where(torch.isfinite(out_sum/out_cnt), out_sum/out_cnt, torch.zeros_like(out_sum))
#             mix.comp_out.copy_((m_fp_out - q_mean_out).to(mix.comp_out.dtype) * comp_out_decay)

#         # =========================
#         # Phase D: propagate (세 보상 모두 on)
#         # =========================
#         next_inps, next_res = [], []
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase D: propagate", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             yj, resj = layer(xj, residual=res_in_list[j], comp_calib = True)
#             next_inps.append(yj.squeeze(0).detach())
#             next_res.append(resj.detach() if torch.is_tensor(resj) else resj)

#         blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#         inps_list, res_in_list = next_inps, next_res

#     model._fp_means = None
#     return model




# @torch.no_grad()
# def mamba_sequential_compensation_tensor(
#     model,
#     device,
#     nsamples,
#     tokenizer,
#     seq_len=2048,
#     dataloader=None,
#     dataset=None,
#     preprocess_fn=None,
#     # in_proj 보상 미사용
#     comp_out_decay=0.1,
#     comp_ssd_decay=0.1,     # ssd 보상 감쇠
#     layers=None,            # 선택 레이어만 보상
# ):
#     import torch
#     from torch import nn
#     from tqdm import tqdm

#     # ==== Per-Tensor 합/카운트(스칼라) ====
#     def _tensor_sum_and_count(y: torch.Tensor):
#         y32 = y.detach().to(torch.float32)
#         return y32.sum(), int(y32.numel())

#     model.eval()
#     blocks = model.backbone.layers
#     fp_means = getattr(model, "_fp_means", None)
#     assert isinstance(fp_means, dict)

#     # --- 캡처 준비(입력 x만) ---
#     model.backbone.embedding = model.backbone.embedding.to(device)
#     blocks[0] = blocks[0].to(device)

#     if dataloader is None and (dataset is None or preprocess_fn is None):
#         from datasets import load_dataset
#         dataset = load_dataset("monology/pile-uncopyrighted",
#                                data_files="val.jsonl.zst", split="train")
#         def _preprocess(data, tokenizer, max_tokens, device):
#             return tokenizer(data["text"], return_tensors="pt",
#                              truncation=True, max_length=max_tokens
#                             ).input_ids.to(device)
#         from functools import partial
#         preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
#                                 max_tokens=seq_len, device=device)

#     inps_list = []
#     class Catcher(nn.Module):
#         def __init__(self, mod): super().__init__(); self.mod = mod
#         def forward(self, x, *args, **kwargs):
#             inps_list.append(x.squeeze(0).detach() if (x.dim()==3 and x.size(0)==1) else x.detach())
#             raise ValueError
#     blocks[0] = Catcher(blocks[0])

#     if dataloader is not None:
#         it = iter(dataloader)
#         for _ in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             batch = next(it)
#             try:
#                 xb = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 model(xb.to(device))
#             except ValueError:
#                 pass
#     else:
#         for i in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             x = preprocess_fn(dataset[i])
#             try: model(x)
#             except ValueError: pass

#     blocks[0] = blocks[0].mod
#     blocks[0] = blocks[0].cpu()
#     model.backbone.embedding = model.backbone.embedding.cpu()
#     torch.cuda.empty_cache()
#     assert len(inps_list) == nsamples

#     # residual 초기화(모델이 residual을 반환하는 인터페이스일 경우 사용)
#     res_in_list = [None] * nsamples

#     # --- 레이어 선택 집합 ---
#     L = len(model.backbone.layers)
#     def _norm_idxs(idxs):
#         s=set()
#         for k in idxs:
#             if k<0: k=L+k
#             if 0<=k<L: s.add(int(k))
#         return s

#     if layers is None:
#         target_layers = set(range(L))
#     else:
#         target_layers = _norm_idxs(list(layers) if not isinstance(layers, (list, tuple)) else layers)

#     # --- 레이어 순차 보상 ---
#     for li in tqdm(range(L), desc="[Layer] Sequential compensation", leave=True):
#         layer = blocks[li].to(device)
#         mix = layer.mixer
#         mix.compensation = True

#         # 대상이 아니면 그냥 전파
#         if li not in target_layers:
#             next_inps, next_res = [], []
#             for j in range(nsamples):
#                 xj = inps_list[j];  xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 yj, resj = layer(xj, residual=res_in_list[j])
#                 next_inps.append(yj.squeeze(0).detach())
#                 next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
#             blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#             inps_list, res_in_list = next_inps, next_res
#             continue

#         # ===== 보상 버퍼 준비 =====
#         # in_proj 보상은 사용하지 않으므로 comp_in은 0 버퍼로 고정
#         if getattr(mix, "comp_in", None) is None:
#             mix.register_buffer("comp_in", torch.zeros(
#                 mix.in_proj.out_features, dtype=torch.float16, device=device))
#         else:
#             mix.comp_in.zero_()

#         if getattr(mix, "ssd_comp", None) is None:
#             mix.register_buffer("ssd_comp", torch.zeros(
#                 mix.d_inner, dtype=torch.float16, device=device))
#         if getattr(mix, "comp_out", None) is None:
#             mix.register_buffer("comp_out", torch.zeros(
#                 mix.out_proj.out_features, dtype=torch.float16, device=device))

#         # fp 기준 평균 (Per-Tensor 스칼라여야 일관적)
#         m_fp_ssd  = fp_means.get(f"L{li}/ssd_out",      None)
#         m_fp_out  = fp_means.get(f"L{li}/out_proj_out", None)
#         if m_fp_ssd is not None: m_fp_ssd = m_fp_ssd.to(device=device, dtype=torch.float32)
#         if m_fp_out is not None: m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)

#         # 모든 보상 초기화
#         mix.comp_in.zero_()
#         mix.ssd_comp.zero_()
#         mix.comp_out.zero_()

#         # =========================
#         # Phase B: ssd_comp 추정 (per-tensor)
#         # =========================
#         ssd_sum=None; ssd_cnt=0
#         if hasattr(mix, "ssd_out_act") and mix.ssd_out_act is not None and (m_fp_ssd is not None):
#             def ssd_hook(_m,_i,out):
#                 nonlocal ssd_sum, ssd_cnt
#                 s, n = _tensor_sum_and_count(out)
#                 ssd_sum = s if ssd_sum is None else (ssd_sum + s)
#                 ssd_cnt += n
#             hB = mix.ssd_out_act.register_forward_hook(ssd_hook)
#             for j in tqdm(range(nsamples), desc=f"[L{li}] Phase B: ssd_out", leave=False):
#                 xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 _ = layer(xj, residual=res_in_list[j], comp_calib=True)
#             hB.remove()
#             if ssd_cnt > 0:
#                 q_mean_ssd = torch.nan_to_num(ssd_sum / ssd_cnt, nan=0.0, posinf=0.0, neginf=0.0)
#                 val = ((m_fp_ssd - q_mean_ssd).to(torch.float32) * comp_ssd_decay).to(mix.ssd_comp.dtype)
#                 mix.ssd_comp.fill_(val)  # 스칼라 → 전체 채우기

#         # =========================
#         # Phase C: comp_out 추정 (per-tensor)
#         # =========================
#         out_sum=None; out_cnt=0
#         def out_hook(_m,_i,out):
#             nonlocal out_sum,out_cnt
#             s, n = _tensor_sum_and_count(out)
#             out_sum = s if out_sum is None else (out_sum + s)
#             out_cnt += n
#         hC = mix.out_proj.register_forward_hook(out_hook)
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase C: out_proj", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             _ = layer(xj, residual=res_in_list[j], comp_calib=True)
#         hC.remove()
#         if (m_fp_out is not None) and (out_cnt > 0):
#             q_mean_out = torch.nan_to_num(out_sum / out_cnt, nan=0.0, posinf=0.0, neginf=0.0)
#             val = ((m_fp_out - q_mean_out).to(torch.float32) * comp_out_decay).to(mix.comp_out.dtype)
#             mix.comp_out.fill_(val)  # 스칼라 → 전체 채우기

#         # =========================
#         # Phase D: propagate (ssd_comp + comp_out on, comp_in=0)
#         # =========================
#         next_inps, next_res = [], []
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase D: propagate", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             yj, resj = layer(xj, residual=res_in_list[j], comp_calib=True)
#             next_inps.append(yj.squeeze(0).detach())
#             next_res.append(resj.detach() if torch.is_tensor(resj) else resj)

#         blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#         inps_list, res_in_list = next_inps, next_res

#     model._fp_means = None
#     return model


# @torch.no_grad()
# def mamba_sequential_compensation_ssdout(
#     model,
#     device,
#     nsamples,
#     tokenizer,
#     seq_len=2048,
#     dataloader=None,
#     dataset=None,
#     preprocess_fn=None,
#     # comp_in_decay 제거됨 (in_proj 보상 미사용)
#     comp_out_decay=0.1,
#     comp_ssd_decay=0.1,     # ssd 보상 감쇠
#     layers=None,            # 선택 레이어만 보상
# ):
#     import torch
#     from torch import nn
#     from tqdm import tqdm

#     def _batch_sum_and_count(y: torch.Tensor):
#         if y.dim() == 3: return y.sum(dim=(0,1)).float(), y.shape[0]*y.shape[1]
#         if y.dim() == 2: return y.sum(dim=0).float(), y.shape[0]
#         raise RuntimeError(f"Unexpected shape {tuple(y.shape)}")

#     model.eval()
#     blocks = model.backbone.layers
#     fp_means = getattr(model, "_fp_means", None)
#     assert isinstance(fp_means, dict)

#     # --- 캡처 준비(입력 x만) ---
#     model.backbone.embedding = model.backbone.embedding.to(device)
#     blocks[0] = blocks[0].to(device)

#     if dataloader is None and (dataset is None or preprocess_fn is None):
#         from datasets import load_dataset
#         dataset = load_dataset("monology/pile-uncopyrighted",
#                                data_files="val.jsonl.zst", split="train")
#         def _preprocess(data, tokenizer, max_tokens, device):
#             return tokenizer(data["text"], return_tensors="pt",
#                              truncation=True, max_length=max_tokens
#                             ).input_ids.to(device)
#         from functools import partial
#         preprocess_fn = partial(_preprocess, tokenizer=tokenizer,
#                                 max_tokens=seq_len, device=device)

#     inps_list = []
#     class Catcher(nn.Module):
#         def __init__(self, mod): super().__init__(); self.mod = mod
#         def forward(self, x, *args, **kwargs):
#             inps_list.append(x.squeeze(0).detach() if (x.dim()==3 and x.size(0)==1) else x.detach())
#             raise ValueError
#     blocks[0] = Catcher(blocks[0])

#     if dataloader is not None:
#         it = iter(dataloader)
#         for _ in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             batch = next(it)
#             try:
#                 xb = batch[0] if isinstance(batch, (list, tuple)) else batch
#                 model(xb.to(device))
#             except ValueError:
#                 pass
#     else:
#         for i in tqdm(range(nsamples), desc="[Init] Capture inputs", leave=False):
#             x = preprocess_fn(dataset[i])
#             try: model(x)
#             except ValueError: pass

#     blocks[0] = blocks[0].mod
#     blocks[0] = blocks[0].cpu()
#     model.backbone.embedding = model.backbone.embedding.cpu()
#     torch.cuda.empty_cache()
#     assert len(inps_list) == nsamples

#     # residual 초기화(모델이 residual을 반환하는 인터페이스일 경우 사용)
#     res_in_list = [None] * nsamples

#     # --- 레이어 선택 집합 ---
#     L = len(model.backbone.layers)
#     def _norm_idxs(idxs):
#         s=set()
#         for k in idxs:
#             if k<0: k=L+k
#             if 0<=k<L: s.add(int(k))
#         return s

#     if layers is None:
#         target_layers = set(range(L))
#     else:
#         target_layers = _norm_idxs(list(layers) if not isinstance(layers, (list, tuple)) else layers)

#     # --- 레이어 순차 보상 ---
#     for li in tqdm(range(L), desc="[Layer] Sequential compensation", leave=True):
#         layer = blocks[li].to(device)
#         mix = layer.mixer
#         mix.compensation = True

#         # 대상이 아니면 그냥 전파
#         if li not in target_layers:
#             next_inps, next_res = [], []
#             for j in range(nsamples):
#                 xj = inps_list[j];  xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 yj, resj = layer(xj, residual=res_in_list[j])
#                 next_inps.append(yj.squeeze(0).detach())
#                 next_res.append(resj.detach() if torch.is_tensor(resj) else resj)
#             blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#             inps_list, res_in_list = next_inps, next_res
#             continue

#         # ===== 보상 버퍼 준비 =====
#         # in_proj 보상은 사용하지 않으므로 comp_in은 0 버퍼로 고정
#         # if getattr(mix, "comp_in", None) is None:
#         #     mix.register_buffer("comp_in", torch.zeros(
#         #         mix.in_proj.out_features, dtype=torch.float16, device=device))
#         # else:
#         #     mix.comp_in.zero_()

#         if getattr(mix, "ssd_comp", None) is None:
#             mix.register_buffer("ssd_comp", torch.zeros(
#                 mix.d_inner, dtype=torch.float16, device=device))
#         if getattr(mix, "comp_out", None) is None:
#             mix.register_buffer("comp_out", torch.zeros(
#                 mix.out_proj.out_features, dtype=torch.float16, device=device))

#         # fp 기준 평균
#         m_fp_ssd  = fp_means.get(f"L{li}/ssd_out",      None)
#         m_fp_out  = fp_means.get(f"L{li}/out_proj_out", None)
#         if m_fp_ssd is not None: m_fp_ssd = m_fp_ssd.to(device=device, dtype=torch.float32)
#         if m_fp_out is not None: m_fp_out = m_fp_out.to(device=device, dtype=torch.float32)

#         # 모든 보상 초기화
#         # mix.comp_in.zero_()
#         mix.ssd_comp.zero_()
#         mix.comp_out.zero_()

#         # =========================
#         # Phase B: ssd_comp 추정
#         # (comp_in=0, ssd_comp/off, comp_out/off)
#         # =========================
#         ssd_sum=None; ssd_cnt=0
#         if hasattr(mix, "ssd_out_act") and mix.ssd_out_act is not None and (m_fp_ssd is not None):
#             def ssd_hook(_m,_i,out):
#                 nonlocal ssd_sum, ssd_cnt
#                 s, n = _batch_sum_and_count(out)
#                 ssd_sum = s if ssd_sum is None else (ssd_sum+s)
#                 ssd_cnt += int(n)
#             hB = mix.ssd_out_act.register_forward_hook(ssd_hook)
#             for j in tqdm(range(nsamples), desc=f"[L{li}] Phase B: ssd_out", leave=False):
#                 xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#                 _ = layer(xj, residual=res_in_list[j], comp_calib=True)
#             hB.remove()
#             if ssd_cnt > 0:
#                 q_mean_ssd = torch.where(torch.isfinite(ssd_sum/ssd_cnt), ssd_sum/ssd_cnt, torch.zeros_like(ssd_sum))
#                 mix.ssd_comp.copy_((m_fp_ssd - q_mean_ssd).to(mix.ssd_comp.dtype) * comp_ssd_decay)

#         # =========================
#         # Phase C: comp_out 추정
#         # (comp_in=0, ssd_comp ON, comp_out/off)
#         # =========================
#         out_sum=None; out_cnt=0
#         def out_hook(_m,_i,out):
#             nonlocal out_sum,out_cnt
#             s, n = _batch_sum_and_count(out)
#             out_sum = s if out_sum is None else (out_sum+s)
#             out_cnt += int(n)
#         hC = mix.out_proj.register_forward_hook(out_hook)
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase C: out_proj", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             _ = layer(xj, residual=res_in_list[j], comp_calib=True)
#         hC.remove()
#         if (m_fp_out is not None) and (out_cnt > 0):
#             q_mean_out = torch.where(torch.isfinite(out_sum/out_cnt), out_sum/out_cnt, torch.zeros_like(out_sum))
#             mix.comp_out.copy_((m_fp_out - q_mean_out).to(mix.comp_out.dtype) * comp_out_decay)

#         # =========================
#         # Phase D: propagate (ssd_comp + comp_out on, comp_in=0)
#         # =========================
#         next_inps, next_res = [], []
#         for j in tqdm(range(nsamples), desc=f"[L{li}] Phase D: propagate", leave=False):
#             xj = inps_list[j]; xj = xj.unsqueeze(0) if xj.dim()==2 else xj
#             yj, resj = layer(xj, residual=res_in_list[j], comp_calib=True)
#             next_inps.append(yj.squeeze(0).detach())
#             next_res.append(resj.detach() if torch.is_tensor(resj) else resj)

#         blocks[li] = layer.cpu(); torch.cuda.empty_cache()
#         inps_list, res_in_list = next_inps, next_res

#     model._fp_means = None
#     return model

