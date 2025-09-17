import json
from utils import (
    build_mamba_and_tokenizer, 
    set_deterministic, 
    parse_options
)

import logging
import sys
import os
from quamba.eval_utils import eval_mamba_few_shot, eval_mamba_generation, evaluate_ppl
from quamba.modelutils_mamba import quantize_model_mamba
from visualizer import register_basic_hooks, plot_tensor,plot_dim_stats
import types, torch.nn as nn
def main(args):    
    model_name = args.model.lower().split('/')[-1]
    model_type = model_name.split('-')[0] # Assume that the models name is like "model_type-<model_size, model version>"
    assert model_name != None, "Please check the model path."
    logging.info(f"Creating Model:{model_name}")
    model, tokenizer, is_quamba = build_mamba_and_tokenizer(args, model_type)


    
    ###########################
    if args.plot:
        model.config.use_cache = False                   # (SSD cache X)
        register_basic_hooks(model,
                            batch_idx = 0,
                            head_idx  = None,   # 특정 head → 숫자
                            gp_idx    = None,   # 특정 group → 숫자
                            verbose   = False)
    ####################################################################



    # for i, blk in enumerate(model.backbone.layers):
    #     print(f"{i:02d} │ {blk.__class__.__name__}")

    # # 2) 타입·파라미터 수도 같이 보고 싶다면
    # for i, blk in enumerate(model.backbone.layers):
    #     n_param = sum(p.numel() for p in blk.parameters())
    #     print(f"{i:02d} │ {blk.__class__.__name__} │ {n_param/1e6:6.2f} M params")

    # # 3) ‘repr’(모듈 요약)까지 세부적으로
    # for i, blk in enumerate(model.backbone.layers):
    #     print(f"\n──── Layer {i} ────────────────────────────")
    #     print(blk)          # nn.Module 의 __repr__ 출력
    ##################
    logs = {}
    
    if args.quantize:
        """
        Start evaluating Quantized Models from here
        """
        if not is_quamba:
            model = quantize_model_mamba(model, model_type, tokenizer, "cuda", args)
    else:
        """
        Evaluate the non-quantized models
        """
        logging.info(f"Evaluating the performance of fp16 model")
    model.eval()

    logs = {}
    if args.eval_ppl:
        logging.info(f"Evaluating ppl result (quantized), dataset: {args.ppl_dataset}")
        ppl_results = evaluate_ppl(model, tokenizer, model_name, batch_size=args.batch_size, device="cuda", dataset=args.ppl_dataset,seq_len=args.ppl_seq_len)
        logs['ppl'] = ppl_results
    if args.eval_zero_shot:
        logging.info(f"Evaluating result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_few_shot(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if args.eval_few_shot:
        logging.info(f"Evaluating {args.fewshot}-shot result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_few_shot(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            fewshot=args.fewshot,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if args.eval_generation:
        logging.info(f"Evaluating generation result using lm_eval (quantized), task(s): {args.task_list}")
        lm_eval_results = eval_mamba_generation(
            model, tokenizer, 
            model_type=model_type,
            task_list=args.task_list, 
            batch_size=args.batch_size,
            fewshot=args.fewshot,
            limit=100 if args.testing else None
        )
        logs['lm_eval'] = lm_eval_results['results']
    if not args.eval_ppl and not args.eval_zero_shot and not args.eval_few_shot and not args.eval_generation:
        logging.warning("No task to run with, try `--eval_ppl`, `--eval_zero_shot`, `--eval_generation`, `--eval_few_shot --fewshot n`?")
        
    if args.log_dir:
        logs['args'] = vars(args)
        os.makedirs(args.log_dir, exist_ok=True)
        if args.quantize:
            log_name = f"{model_name}" if is_quamba else f"{model_name}_w{args.w_bits}a{args.a_bits}"
            log_paths = os.path.join(args.log_dir, f"{log_name}.json")
        else:
            log_paths = os.path.join(args.log_dir, f"{model_name}_fp16.json")
        with open(log_paths, 'a') as fp:
            logging.info(f"Saving result to {log_paths}")
            json.dump(logs, fp, indent=4)
    if args.plot:
        for t in ["Z","X","B","C","dt","OUT"]:
            plot_tensor(t, mode="3d",flatten=False)               # 또는 "heat"
        for t in ["X", "Z",  "OUT", "B", "C","dt"]:
            plot_dim_stats(t)  
if __name__ =='__main__':    
    # set_deterministic(1234)
    set_deterministic(9997)
    args = parse_options()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
    main(args)

