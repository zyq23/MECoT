# import torch
# from transformers import AutoTokenizer, AutoModel
# from peft import PeftModel
# import os
#
# # 配置路径（确保与你训练脚本中的路径一致）
# BASE_MODEL_PATH = "/data/lzk/zyq_projects/mental_health_debate_framework/chatglm2-6b"
# LORA_PATH = "./output/cbt_finetuned_v2"
#
# # 这里的 System Prompt 必须与你训练时 data_utils.py 中的完全一致
# CBT_SYSTEM_PROMPT = (
#     """你是一位专业的心理咨询师，精通认知行为疗法（CBT）。
# 你的任务不是单纯的安慰，而是通过对话引导来访者识别并改变其负面的思维模式（认知扭曲）和行为习惯。
#
# 请遵循以下 CBT 治疗阶段进行回复：
# 1. **建立关系与评估**：表现出高度的共情，接纳来访者的情绪，建立信任。
# 2. **识别认知扭曲**：敏锐地捕捉来访者话语中的“非黑即白”、“灾难化思维”、“贴标签”等非理性信念。
# 3. **苏格拉底式提问**：不要直接给建议，而是通过提问（例如：“有什么证据支持你的这个想法吗？”）引导来访者自我反思。
# 4. **行为实验与作业**：在对话中后期，提供具体的、可操作的小实验（如行为激活、记录情绪日记），帮助来访者在现实中验证新思维。
#
# 注意：保持语气温暖、专业、不评判。回复应简练且具有引导性。"""
# )
#
#
# def load_cbt_model():
#     print("正在加载底座模型 (Base Model)...")
#     tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
#     model = AutoModel.from_pretrained(
#         BASE_MODEL_PATH,
#         trust_remote_code=True,
#         device_map="auto",
#         torch_dtype=torch.float16
#     )
#
#     print("正在挂载 CBT 微调权重 (LoRA Adapter)...")
#     # 加载你刚刚训练好的 LoRA 权重
#     model = PeftModel.from_pretrained(model, LORA_PATH)
#     model = model.eval()  # 切换到推理模式
#     return model, tokenizer
#
#
# def main():
#     model, tokenizer = load_cbt_model()
#
#     print("\n" + "=" * 50)
#     print("CBT 心理咨询模型已就绪。输入 'exit' 退出对话。")
#     print("=" * 50)
#
#     history = []
#
#     while True:
#         user_input = input("\n[来访者]: ")
#         if user_input.strip().lower() == 'exit':
#             break
#
#         # 处理第一轮对话，注入 System Prompt 引导模型角色
#         if len(history) == 0:
#             query = f"{CBT_SYSTEM_PROMPT}\n\n[Round 1]\n\n问：{user_input}\n\n答："
#         else:
#             query = user_input
#
#         # 使用 ChatGLM 的 chat 接口进行多轮对话
#         # 注意：这里的 history 会自动维护，不需要手动拼接 Round 标记
#         response, history = model.chat(tokenizer, query, history=history)
#
#         print(f"\n[CBT 咨询师]: {response}")
#
#
# if __name__ == "__main__":
#     main()


import torch
from transformers import AutoTokenizer, AutoModel

# 指向你合并后保存的完整权重目录
MODEL_PATH = "./output/cbt_chatglm2-6b_merged"

# 保持与训练一致的 System Prompt，以确保模型进入正确的咨询状态
CBT_SYSTEM_PROMPT = (
    """你是一位专业的心理咨询师，精通认知行为疗法（CBT）。
你的任务不是单纯的安慰，而是通过对话引导来访者识别并改变其负面的思维模式（认知扭曲）和行为习惯。

请遵循以下 CBT 治疗阶段进行回复：
1. **建立关系与评估**：表现出高度的共情，接纳来访者的情绪，建立信任。
2. **识别认知扭曲**：敏锐地捕捉来访者话语中的“非黑即白”、“灾难化思维”、“贴标签”等非理性信念。
3. **苏格拉底式提问**：不要直接给建议，而是通过提问（例如：“有什么证据支持你的这个想法吗？”）引导来访者自我反思。
4. **行为实验与作业**：在对话中后期，提供具体的、可操作的小实验（如行为激活、记录情绪日记），帮助来访者在现实中验证新思维。

注意：保持语气温暖、专业、不评判。回复应简练且具有引导性。"""
)


def main():
    print(f"正在从 {MODEL_PATH} 加载合并后的 CBT 模型...")

    # 1. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 2. 加载合并后的完整模型
    # 使用 .half() 开启半精度推理，.cuda() 将模型移至显卡
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"
    ).half().cuda()

    model = model.eval()

    print("\n" + "=" * 50)
    print("✅ CBT 心理咨询大模型（完整合并版）已就绪。")
    print("对话开始，输入 'exit' 可退出。")
    print("=" * 50)

    history = []

    while True:
        user_input = input("\n[来访者]: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break

        # 第一轮对话时注入 System Prompt
        if len(history) == 0:
            query = f"{CBT_SYSTEM_PROMPT}\n\n[Round 1]\n\n问：{user_input}\n\n答："
        else:
            query = user_input

        # 使用 ChatGLM2 官方 chat 接口
        # 该接口会自动处理 history 中的上下文逻辑
        try:
            response, history = model.chat(tokenizer, query, history=history)
            print(f"\n[CBT 咨询师]: {response}")
        except Exception as e:
            print(f"\n❌ 推理出错: {e}")


if __name__ == "__main__":
    main()