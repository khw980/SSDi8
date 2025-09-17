
mkdir -p logs

# 배치 사이즈 리스트
BATCH_SIZES=(16)

# 모델 리스트
MODELS=(
    "state-spaces/mamba2-1.3B"
    "state-spaces/mamba2-2.7B"
    "pretrained_models/mamba2-8b-converted"
)

for model in "${MODELS[@]}"; do
    modelname=$(basename "$model")

    for BS in "${BATCH_SIZES[@]}"; do
        echo "==== Running ${model} with batch_size=${BS} ===="

        python -W ignore main.py "$model" \
            --batch_size "$BS" \
            --eval_ppl \
            --eval_zero_shot \
            --task_list lambada_openai,hellaswag,arc_easy,arc_challenge,piqa,winogrande \
            --quantize \
            --log_dir logs \
            --w_bits 4 \
            --a_bits 8 \
            --apply_hadamard \
            --apply_gptq \
            --compensation \
            --comp_ssd_decay 0.16 \
            --comp_out_decay 0.16 \
            2>&1 | tee "logs/48SSDim${modelname}${BS}.log"
    done
done
