import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
import os

# 1. 设置路径
BASE_MODEL_PATH = "/data/lzk/zyq_projects/mental_health_debate_framework/chatglm2-6b"
# 注意：这里指向包含 adapter_config.json 的具体子目录，即你上一条报错中确认的路径
LORA_PATH = "./output/humanistic_model"
SAVE_PATH = "./output/humanistic_chatglm2-6b_merged"


def merge_and_save():
    print(f"正在加载底座模型: {BASE_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    # 加载底座模型，建议先不使用 device_map="auto"，手动指定设备以确保合并过程稳定
    base_model = AutoModel.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"  # 合并操作对显存要求高，如果显存不够可以先在 CPU 上合并
    )

    print(f"正在加载 LoRA 权重: {LORA_PATH}...")
    # 加载 LoRA 适配器
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("正在进行权重合并 (Merging)...")
    # 核心步骤：将 LoRA 层参数加回到底座模型参数中
    merged_model = model.merge_and_unload()

    print(f"正在保存完整模型到: {SAVE_PATH}...")
    # 保存合并后的完整模型和分词器
    merged_model.save_pretrained(SAVE_PATH, max_shard_size="2GB")  # 分片保存，方便后续加载
    tokenizer.save_pretrained(SAVE_PATH)

    print("\n✅ 合并完成！你现在可以直接使用此路径加载完整模型，无需再使用 PEFT。")


if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    merge_and_save()