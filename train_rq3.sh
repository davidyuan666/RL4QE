#!/bin/bash

# 询问用户想要运行哪个模型
echo "请选择要训练的模型:"
echo "1) Llama-2-7B with LoRA"
echo "2) T5-base"
echo "3) Qwen-7B (恢复训练)"
echo "4) Llama-2-7B with 4bit quantization"
echo "5) Mistral-7B (快速评估)"
echo "6) 自定义配置"
echo "0) 退出"

read -p "请输入选项 (0-6): " choice

# 设置默认参数
MODEL_NAME=""
MODEL_TYPE=""
USE_LORA=""
USE_8BIT=""
USE_4BIT=""
GRADIENT_ACCUM=""
NUM_EPOCHS=""
MAX_SAMPLES=""
RESUME=""

case $choice in
    1)
        MODEL_NAME="meta-llama/Llama-2-7b-hf"
        USE_LORA="--use_lora"
        USE_8BIT="--use_8bit"
        ;;
    2)
        MODEL_NAME="t5-base"
        MODEL_TYPE="--model_type seq2seq_lm"
        USE_LORA="--use_lora"
        ;;
    3)
        MODEL_NAME="Qwen/Qwen-7B"
        RESUME="--resume"
        ;;
    4)
        MODEL_NAME="meta-llama/Llama-2-7b-hf"
        USE_4BIT="--use_4bit"
        GRADIENT_ACCUM="--gradient_accumulation_steps 16"
        ;;
    5)
        MODEL_NAME="mistralai/Mistral-7B-v0.1"
        NUM_EPOCHS="--num_epochs 2"
        MAX_SAMPLES="MAX_SAMPLES=100"
        ;;
    6)
        read -p "输入模型名称: " MODEL_NAME
        read -p "模型类型 (causal_lm/seq2seq_lm, 留空默认causal_lm): " model_type_input
        if [ "$model_type_input" = "seq2seq_lm" ]; then
            MODEL_TYPE="--model_type seq2seq_lm"
        fi
        
        read -p "使用LoRA? (y/n): " lora_choice
        if [ "$lora_choice" = "y" ]; then
            USE_LORA="--use_lora"
        fi
        
        read -p "使用8位量化? (y/n): " bit8_choice
        if [ "$bit8_choice" = "y" ]; then
            USE_8BIT="--use_8bit"
        fi
        
        read -p "使用4位量化? (y/n): " bit4_choice
        if [ "$bit4_choice" = "y" ]; then
            USE_4BIT="--use_4bit"
        fi
        
        read -p "梯度累积步数 (留空为默认): " grad_accum
        if [ -n "$grad_accum" ]; then
            GRADIENT_ACCUM="--gradient_accumulation_steps $grad_accum"
        fi
        
        read -p "训练轮数 (留空为默认): " epochs
        if [ -n "$epochs" ]; then
            NUM_EPOCHS="--num_epochs $epochs"
        fi
        
        read -p "限制样本数量 (留空为使用全部): " samples
        if [ -n "$samples" ]; then
            MAX_SAMPLES="MAX_SAMPLES=$samples"
        fi
        
        read -p "恢复训练? (y/n): " resume_choice
        if [ "$resume_choice" = "y" ]; then
            RESUME="--resume"
        fi
        ;;
    0)
        echo "退出脚本"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

# 构建并执行命令
if [ -n "$MODEL_NAME" ]; then
    CMD="$MAX_SAMPLES python train_rq3.py --model_name \"$MODEL_NAME\" $MODEL_TYPE $USE_LORA $USE_8BIT $USE_4BIT $GRADIENT_ACCUM $NUM_EPOCHS $RESUME"
    echo "执行命令: $CMD"
    eval $CMD
else
    echo "未选择模型，退出"
fi