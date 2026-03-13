import os
import torch
import json
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

# ================= 1. 全局配置 (No-CoT Ablation) =================
CONFIG = {
    # ChatGLM2-6B 基座模型路径
    "model_path": "/data/zyq/mental_health_debate_framework/chatglm2-6b",

    "data_path": "train_data_nocot.json",

    # 输出目录
    "output_dir": "./output_chair_nocot_checkpoints",

    # LoRA 参数
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,

    # 训练参数
    "max_len": 1536,
    "epochs": 5,
    "batch_size": 4,
    "grad_accum": 4,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "seed": 42
}

set_seed(CONFIG["seed"])


# ================= 2. 数据处理函数 =================
def process_func(example, tokenizer, max_len):

    prompt = f"{example['instruction']}\n\n{example['input']}\n\n请回复："
    target = example['output']

    # 编码
    instruction_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    eos_token_id = tokenizer.eos_token_id

    # 拼接
    input_ids = instruction_ids + target_ids + [eos_token_id]

    labels = [-100] * len(instruction_ids) + target_ids + [eos_token_id]

    # 长度截断
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]

    return {"input_ids": input_ids, "labels": labels}


# ================= 3. 主训练流程 =================
def train_main():
    # 禁用 WandB 防止报错或需要登录
    os.environ["WANDB_DISABLED"] = "true"

    print("🚀 初始化 No-CoT (Checkpoint Only) 训练环境...")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. 加载模型
    model = AutoModel.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 开启梯度检查点 (Gradient Checkpointing) 以节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        # 针对 ChatGLM 的线性层进行微调
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 加载并处理数据
    if not os.path.exists(CONFIG["data_path"]):
        raise FileNotFoundError(f"❌ 错误：找不到数据文件: {CONFIG['data_path']}，请先运行 format_data_nocot.py")

    with open(CONFIG["data_path"], 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    # 转换为 Dataset 对象 (不划分验证集，全量训练)
    dataset = Dataset.from_list(data_list)

    # 并行处理数据
    dataset = dataset.map(
        lambda x: process_func(x, tokenizer, CONFIG["max_len"]),
        remove_columns=dataset.column_names,
        num_proc=4
    )

    # 5. 设置训练参数
    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["grad_accum"],
        learning_rate=CONFIG["lr"],
        num_train_epochs=CONFIG["epochs"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=CONFIG["weight_decay"],

        # 精度设置
        fp16=False,
        bf16=True,  # 3090推荐 BF16

        # --- 【Checkpoints 保存策略】 ---
        save_strategy="steps",
        # 每 100 步保存一次
        save_steps=100,
        # 最多只保留最新的 3 个存档，防止硬盘爆满
        save_total_limit=3,

        logging_steps=10,
        gradient_checkpointing=True,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_num_workers=0
    )

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        # DataCollator 负责 Batch 内的 Padding
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8
        )
    )

    print("🔥 开始 No-CoT 微调训练...")
    # 如果目录下有 checkpoint，自动尝试断点续训
    trainer.train(resume_from_checkpoint=False)

    # 7. 保存最终模型
    final_save_path = os.path.join(CONFIG["output_dir"], "final_chair_model_nocot")
    trainer.save_model(final_save_path)

    # 显式保存 Tokenizer 和 Config，方便后续推理直接加载
    tokenizer.save_pretrained(final_save_path)
    model.save_pretrained(final_save_path)

    print(f"✅ 训练完成！最终模型已保存至: {final_save_path}")


if __name__ == "__main__":
    train_main()