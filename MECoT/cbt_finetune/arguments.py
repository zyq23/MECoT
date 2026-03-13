# arguments.py
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/data/lzk/zyq_projects/mental_health_debate_framework/chatglm2-6b",
        metadata={"help": "底座模型权重的路径或 HuggingFace ID"}
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "LoRA 的秩 (Rank)"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha 参数"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout 参数"}
    )


@dataclass
class DataArguments:
    data_dir: str = field(
        default="../data_cbt",
        metadata={"help": "包含多个 json 文件的目录路径"}
    )
    max_source_length: int = field(
        default=1024,
        metadata={"help": "输入序列（System+History+Query）的最大长度"}
    )
    max_target_length: int = field(
        default=512,
        metadata={"help": "输出序列（Response）的最大长度"}
    )


@dataclass
class TrainingArgumentsCustom:
    # 这里主要是为了方便IDE提示，实际继承自 transformers.TrainingArguments
    output_dir: str = field(default="output/cbt_counselor")