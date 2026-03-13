import os
import torch
import json
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# ================= 1. 全局配置 (RTX 3090 专属优化) =================
CONFIG = {
    # 路径配置
    "model_path": "/data/zyq/mental_health_debate_framework/chatglm2-6b",
    "data_path": "train_data_cleaned.json",
    "output_dir": "./output_chair_restart",  # 新的输出目录

    # LoRA 参数
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,  # 替代 NEFTune 的抗过拟合作用

    # 训练参数
    "max_len": 1536,
    "epochs": 5,  #  5 轮
    "batch_size": 4,  # 3090 显存较大，可设为 4
    "grad_accum": 4,  # 等效 Batch Size = 16
    "lr": 5e-5,  # LoRA 学习率
    "weight_decay": 0.01,  # L2 正则化
    "seed": 42
}

# 固定随机种子
set_seed(CONFIG["seed"])


# ================= 2. 数据处理 =================
def process_func(example, tokenizer, max_len):
    # Prompt 结构： Instruction + Input -> Output
    prompt = f"{example['instruction']}\n\n{example['input']}\n\n请回复："
    target = example['output']

    instruction_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id

    input_ids = instruction_ids + target_ids + [eos_token_id]
    labels = [-100] * len(instruction_ids) + target_ids + [eos_token_id]

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}


# ================= 3. 主训练流程 =================
def train_main():
    # 禁用 WandB
    os.environ["WANDB_DISABLED"] = "true"

    print("🚀 初始化训练环境 (BF16 Stable Mode)...")

    # 1. 加载模型与分词器
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


    model = AutoModel.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16  # <--- 这里的改动很关键
    )

    # 开启梯度检查点 (省显存)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 2. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    model = get_peft_model(model, peft_config)
    print("⚠️ 已禁用 NEFTune 以解决与梯度检查点的冲突，系统将更加稳定。")
    model.print_trainable_parameters()

    # 3. 加载数据
    if not os.path.exists(CONFIG["data_path"]):
        raise FileNotFoundError(f"找不到数据文件: {CONFIG['data_path']}")

    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)

    dataset = dataset.map(
        lambda x: process_func(x, tokenizer, CONFIG["max_len"]),
        remove_columns=dataset.column_names,
        num_proc=1
    )

    # 4. 设置训练参数 (稳定性拉满)
    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        num_train_epochs=CONFIG["epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=CONFIG["weight_decay"],

        # 【关键】开启 BF16，关闭 FP16
        fp16=False,
        bf16=True,

        # 梯度裁剪 (防止爆炸)
        max_grad_norm=1.0,

        # 【关键】高频存档策略
        save_strategy="steps",
        save_steps=500,  # 每 500 步存一次
        save_total_limit=3,

        logging_steps=10,
        gradient_checkpointing=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8
        )
    )

    print("🔥 开始全量训练...")
    trainer.train()

    # 5. 保存最终模型
    save_path = os.path.join(CONFIG["output_dir"], "final_chair_model")
    trainer.save_model(save_path)
    print(f"✅ 训练圆满完成！模型已保存至: {save_path}")


if __name__ == "__main__":
    train_main()

