import os
import torch
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# ================= 配置区域 =================
# 指向 No-CoT 模型路径
LORA_PATH = "./output_chair_nocot/final_chair_model_nocot"
BASE_MODEL_PATH = "/data/zyq/mental_health_debate_framework/chatglm2-6b"

# 【必须】与 format_data_nocot.py 中的 Prompt 保持绝对一致
SYSTEM_PROMPT_NOCOT = """你不是普通的心理咨询师，你是**"多专家会诊系统"的首席决策者（主席智能体）**。
你的核心能力是**深度融合**认知行为疗法（CBT）与人本主义疗法，为用户提供既有温度又有逻辑的专业支持。

请直接基于用户的诉求生成回复，无需输出思考过程。

**回复要求：**
- **自然融合**：回复必须是一段自然、流畅的对话。
- **人本基调**：首先以温暖、接纳的态度回应用户情绪（人本主义）。
- **CBT引导**：随后平滑地引入认知视角的引导或提问（CBT），不要生硬转折。
- **结尾**：给予支持或温和的行动邀请。

【重要约束】
- 禁止复读：如果用户回复简短，请推进对话。
- 拒绝空洞：回复必须针对用户当前的具体困扰。
"""


class ChairInferenceNoCoT:
    def __init__(self):
        print("🚀 正在加载基座模型 (ChatGLM2-6B)...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        print(f"🔄 正在挂载 No-CoT LoRA 权重: {LORA_PATH} ...")
        self.model = PeftModel.from_pretrained(model, LORA_PATH)
        self.model.eval()
        self.history = []
        print("✅ 主席智能体 (无思维链版) 就绪！")

    def build_prompt(self, user_input):
        prompt = f"{SYSTEM_PROMPT_NOCOT}\n\n"
        if self.history:
            prompt += "【对话历史摘要】：\n"
            for role, content in self.history[-4:]:
                tag = "来访者" if role == "client" else "咨询师"
                prompt += f"{tag}：{content}\n"
        prompt += f"\n【当前来访者诉求】：{user_input}\n\n请回复："
        return prompt

    def chat(self, user_input):
        prompt = self.build_prompt(user_input)

        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            max_length=2048,
            top_p=0.9,
            temperature=0.7,
            do_sample=True
        )

        # 直接清理多余空白即可，无需正则解析
        reply = response.strip()

        # 更新历史
        self.history.append(("client", user_input))
        self.history.append(("counselor", reply))

        return reply


def main():
    agent = ChairInferenceNoCoT()
    print("\n" + "=" * 50)
    print("💡 心理咨询室 (消融对照组) 已开启")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n👤 来访者: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]:
                break

            reply = agent.chat(user_input)
            print(f"\n👩‍⚕️ [主席-NoCoT]:\n{reply}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()