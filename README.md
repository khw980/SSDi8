# SSDi8



- 🔧 Supports W4A8 / W8A8 for Mamba2



## Setup

### Hardware Requirements
- NVIDIA GPU Ampere architecture or above

### Software Requirements
- CUDA 12.1 or above
- CMAKE version 3.22.1 or above

### Clone Quamba
- Clone the repository with all submodules:
```bash
git clone --recurse-submodules git@github.com:enyac-group/Quamba.git
# or
cd Quamba
###Need SSH key
git submodule update --init --recursive
```

- Create Quamba conda environment
```bash
cd SSDi8
conda create -n quamba python=3.10
conda activate quamba
pip install -r requirements.txt
```

### Build 3rd-party Libraries

- Install `fast-hadamard-transform`:
```bash
# set force build to include 12N, 40N from the newer commit
export FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE
pip install 3rdparty/fast-hadamard-transform
```

- Install `lm-evaluation-harness`:
```bash
# lm_eval-0.4.2 word2number-1.1
pip install 3rdparty/lm-evaluation-harness
``````

- Install mamba
```bash
# set force build to use the commit for Quamba
export MAMBA_FORCE_BUILD=TRUE
pip install 3rdparty/mamba
```

- Install CUTLASS
```bash
# cmake version >= 3.22.1
bash build_cutlass.sh
```

- Install Megatron-LM
```bash
pip install -e 3rdparty/Megatron-LM
# Not sure why Megatron-LM will force to install pytorch 2.6.0+cu124,
# run `pip install -r requirements.txt` again if necessary
```

### Build Quamba
```bash
pip install .

## Need causal-conv1d-1.5.0.post8
```
- Triton update
```bash
pip uninstall -y triton
pip install triton==3.4.0
```

## Download Models
```bash
# huggingface-cli download ut-enyac/quamba2-{size}-{precision}  --local-dir pretrained_models/ut-enyac/quamba2-{size}-{precision}
huggingface-cli download ut-enyac/quamba2-2.7b-w4a8  --local-dir pretrained_models/ut-enyac/quamba2-2.7b-w4a8
```

## Generate

```bash
python generate.py ut-enyac/quamba2-2.7b-w4a8 --prompt "My cat wrote all this CUDA code for a new language model and" --topp 0.9 --temperature 0.7 --repetition_penalty 1.2 --quantize --cache_graph --pretrained_dir pretrained_models
```

## Evaluate
```bash
bash eval.sh ut-enyac/quamba2-2.7b-w4a8
```


## Mamba2-8B

**[TL;DR]** We provide the 8B model in all precision formats on Hugging Face. To use it, run:
```bash
huggingface-cli download ut-enyac/quamba2-8b-converted-w4a8  --local-dir pretrained_models/ut-enyac/quamba2-8b-converted-w4a8
python main.py ut-enyac/quamba2-8b-converted-w4a8 \
--batch_size 16 \
--eval_zero_shot \
--task_list lambada_openai \
--pretrained_dir ./pretrained_models \
--log_dir logs
```

### Convert Nvidia Mamba2-8B to HuggingFace

Download the checkpoint using `huggingface-cli`
```bash
huggingface-cli download nvidia/mamba2-8b-3t-4k --local-dir ./pretrained_models/mamba2-8b-3t-4k
```
After downloading, you will have the directory `./pretrained_models/mamba2-8b-3t-4k` having a structure like this
```bash
├── latest_checkpointed_iteration.txt
├── mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model (This is tokenizer)
├── README.md
└── release
    └── mp_rank_00
        └── model_optim_rng.pt (This is weights)
```
+ Run the conversion scripts to get the model directory
```bash
python convert_mamba2_8b_to_hf.py \
./pretrained_models/mamba2-8b-3t-4k/release/mp_rank_00/model_optim_rng.pt \
./pretrained_models/mamba2-8b-3t-4k/mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model \
--model_save_path ./pretrained_models/mamba2-8b-converted
```

### Quantize and Evaluate Mamba2-8B

After running, you will see a directory called `mamba2-8b-converted` has been created. Then you can run it with evaluation, profiling as the instructions above. However, it requires at least *24GB* memory on the GPU to quantize the Mamba2-8b model.

For example:
```bash
# use the `--pretrained_dir` flag to store the quantized model
python main.py pretrained_models/mamba2-8b-converted \
--batch_size 16 \
--eval_zero_shot \
--task_list lambada_openai \
--quantize \
--group_heads \
--apply_gptq \
--quantize_embedding \
--quantize_lm_head \
--w_bits 4 \
--a_bits 8
--pretrained_dir ./pretrained_models \
--log_dir logs
``` 



