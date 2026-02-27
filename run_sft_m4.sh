#!/bin/bash
# 适配Apple M4芯片的supervised_finetuning.py执行脚本
# M4芯片使用MPS (Metal Performance Shaders) 后端

# 设置MPS后端
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 单GPU执行（M4不支持多GPU torchrun）
python supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --train_file_dir ./data/finetune \
    --validation_file_dir ./data/finetune \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples 1000 \
    --max_eval_samples 10 \
    --model_max_length 2048 \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --warmup_steps 5 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 13 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-sft-qwen-m4 \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype float16 \
    --device_map cpu \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache

