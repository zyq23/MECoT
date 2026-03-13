import torch
from transformers import AutoTokenizer, AutoModel

# 指向你合并后的完整模型目录
MODEL_PATH = "./output/humanistic_chatglm2-6b_merged"

# 精确详细的人本主义 System Prompt
HUMANISTIC_SYSTEM_PROMPT = (
    "你是一位精通人本主义疗法（Humanistic Therapy）的心理咨询师，深谙卡尔·罗杰斯（Carl Rogers）的咨询理念。"
    "你的核心任务不是以此来解决问题或给出建议，而是通过通过创造一个安全、接纳的氛围，帮助来访者自我探索和成长。"
    "请严格遵循以下原则进行回复："
    "1. **共情理解 (Empathy)**：深入设身处地地去体会来访者的内心感受，并精准地将这些情绪反馈给对方（例如：“听起来你感到...”）。"
    "2. **无条件积极关注 (Unconditional Positive Regard)**：无论来访者表达什么，都给予完全的接纳和尊重，不评判、不指责。"
    "3. **真诚一致 (Congruence)**：保持真实和坦诚，用温暖、支持性的语言与来访者连接。"
    "4. **非指导性 (Non-directive)**：避免教导来访者“该怎么做”或分析其思维错误。相信来访者拥有自我治愈的潜能，通过提问引导其向内看（例如：“这对你来说意味着什么？”）。"
    "注意：多使用情感反映技术，聚焦于“此时此刻”的感受，而非过去的逻辑分析。"
)


def main():
    print(f"正在加载合并后的 Humanistic 完整模型: {MODEL_PATH}...")

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto"
    ).half().cuda()

    model = model.eval()

    print("\n" + "=" * 60)
    print("🧘 人本主义大模型已就绪。你可以开始倾诉你的感受。")
    print("提示：输入 'exit' 退出。")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\n[来访者]: ")
        if user_input.strip().lower() in ['exit', 'quit']:
            break

        # 第一轮对话时注入 System Prompt 以对齐模型行为
        if len(history) == 0:
            query = f"{HUMANISTIC_SYSTEM_PROMPT}\n\n[Round 1]\n\n问：{user_input}\n\n答："
        else:
            query = user_input

        # 使用 ChatGLM 官方接口进行推理
        try:
            response, history = model.chat(tokenizer, query, history=history)
            print(f"\n[咨询师]: {response}")
        except Exception as e:
            print(f"\n❌ 推理过程中出现错误: {e}")


if __name__ == "__main__":
    main()