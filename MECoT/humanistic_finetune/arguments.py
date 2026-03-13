# arguments.py
from dataclasses import dataclass, field

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/data/lzk/zyq_projects/mental_health_debate_framework/chatglm2-6b")
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.1)

@dataclass
class DataArguments:
    # 默认指向人本主义数据目录
    data_dir: str = field(default="../data_humanistic")
    max_source_length: int = field(default=1024)
    max_target_length: int = field(default=512)

@dataclass
class TrainingArgumentsCustom:
    # 默认输出目录
    output_dir: str = field(default="output/humanistic_model")