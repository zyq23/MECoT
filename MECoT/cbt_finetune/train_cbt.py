# train_cbt.py
import os
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from peft import get_peft_model, LoraConfig, TaskType
from arguments import ModelArguments, DataArguments
from data_utils import CBTConversationDataset, CBTDataCollator

os.environ["WANDB_DISABLED"] = "true"


# ------------------------------------------------

def main():
    # 1. 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # 如果直接运行脚本，没有传参，则使用默认值
    if len(sys.argv) == 1:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses([
            "--output_dir", "output/cbt_finetuned_v2",
            "--num_train_epochs", "5",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--save_steps", "100",
            "--logging_steps", "10",
            "--learning_rate", "2e-4",
            "--fp16", "True",
            "--remove_unused_columns", "False"  # 必须设为False，防止自定义Dataset字段被过滤
        ])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Training with model: {model_args.model_name_or_path}")
    print(f"Data directory: {data_args.data_dir}")

    # 2. 加载模型与 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",  # 自动分配显卡
        torch_dtype=torch.float16
    )

    # 开启梯度检查点，节省显存
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. 配置 LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        # ChatGLM 的全量线性层，这样效果最好
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. 准备数据
    dataset = CBTConversationDataset(data_args, tokenizer)
    collator = CBTDataCollator(
        tokenizer,
        max_source_len=data_args.max_source_length,
        max_target_len=data_args.max_target_length
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # 6. 开始训练
    print(">>> Start Training...")
    # trainer.train()
    trainer.train(resume_from_checkpoint=True)

    # 7. 保存模型
    print(f">>> Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)

    # 保存 Tokenizer 方便推理时直接加载
    model.save_pretrained(training_args.output_dir)  # 强制在根目录保存 LoRA 权重
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()