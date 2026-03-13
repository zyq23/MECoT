# train_humanistic.py
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
from data_utils import HumanisticDataset, HumanisticCollator

# 禁用 wandb 防止报错
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 1:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses([
            "--output_dir", "output/humanistic_model",
            "--num_train_epochs", "5",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--learning_rate", "2e-4",
            "--save_steps", "100",
            "--fp16", "True",
            "--remove_unused_columns", "False",
            "--logging_steps", "10"
        ])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Training Humanistic Model Base: {model_args.model_name_or_path}")

    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 准备数据
    dataset = HumanisticDataset(data_args, tokenizer)
    collator = HumanisticCollator(
        tokenizer,
        max_source_len=data_args.max_source_length,
        max_target_len=data_args.max_target_length
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(">>> 开始人本主义风格微调...")
    trainer.train()

    # 保存结果
    trainer.save_model(training_args.output_dir)
    # 强制保存 Tokenizer 和 LoRA Config
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()