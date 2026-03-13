import os
import torch
import re
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# ================= 配置区域 =================
# 指向你刚刚训练好的 LoRA 模型路径
LORA_PATH = "./output_chair_final_finish/final_chair_model_completed"
BASE_MODEL_PATH = "/data/zyq/mental_health_debate_framework/chatglm2-6b"

# 必须与训练时的 System Prompt 保持一个字符都不差，以激活特定潜能
SYSTEM_PROMPT = """你不是普通的心理咨询师，你是**"多专家会诊系统"的首席决策者（主席智能体）**。
你的核心能力是**深度融合**认知行为疗法（CBT）与人本主义疗法，为用户提供既有温度又有逻辑的专业支持。

请严格遵循以下步骤进行思考和回复（Thinking Chain）：

1. **第一步：深度分析 (Analysis)**
   在 <analysis> 标签中，你必须先进行多流派视角的内心推演：
   - **情绪诊断**：精准识别用户的核心情绪与强度。
   - **CBT视角**：像 CBT 专家一样思考，识别用户的认知扭曲（如非黑即白、灾难化），并构思苏格拉底式提问或行为作业。
   - **人本视角**：像人本主义专家一样思考，提炼共情要点，给予无条件接纳。

2. **第二步：融合生成 (Response)**
   在 <response> 标签中，基于上述分析生成最终回复。
   
   **回复要求：**
   - **自然融合**：回复必须是一段自然、流畅的对话，**严禁出现“共情接纳：”、“第一步：”等结构性标签**。
   - **人本基调**：首先以温暖、接纳的态度回应用户情绪（人本主义）。
   - **CBT引导**：随后平滑地引入认知视角的引导或提问（CBT），不要生硬转折。
   - **结尾**：给予支持或温和的行动邀请。

【重要约束】
- 禁止复读：如果用户回复简短（如“好的”），请根据上一轮的分析推进对话，不要重复询问身体症状。
- 拒绝空洞：分析必须具体，回复必须针对用户当前的具体困扰。
- 语言多样性：严禁每句话都以“我听到”、“我能感受到”开头。请使用更丰富、自然、口语化的回应方式（如：“确实，这种感觉很难受”、“难怪你会这么想”等）。
- 顺应对话节奏：如果来访者的回复很简短（如表示同意或感谢），你只需给予简短的肯定和下一步的具体指引，切忌长篇大论或反复提及过去的痛点。
"""


class ChairInference:
    def __init__(self):
        print("🚀 正在加载基座模型 (ChatGLM2-6B)...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            BASE_MODEL_PATH,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        print(f"🔄 正在挂载主席智能体 LoRA 权重: {LORA_PATH} ...")
        self.model = PeftModel.from_pretrained(model, LORA_PATH)
        self.model.eval()

        # 历史记录：存储 (Role, Content) 元组
        self.history = []
        print("✅ 主席智能体就绪！")

    def build_prompt(self, user_input):
        """
        构造与训练格式完全一致的 Prompt
        格式：System + History + Input -> (等待模型补全 Output)
        """
        prompt = f"{SYSTEM_PROMPT}\n\n"

        # 拼接历史 (最近 4 轮，防止上下文溢出)
        if self.history:
            prompt += "【对话历史摘要】：\n"
            for role, content in self.history[-4:]:
                # 这里的 role 显示名称要与 format_data.py 里的一致
                tag = "来访者" if role == "client" else "咨询师"
                # 注意：存入历史的应该是纯净的回复，不包含 analysis 标签
                prompt += f"{tag}：{content}\n"

        # 拼接当前输入
        prompt += f"\n【当前来访者诉求】：{user_input}\n\n请回复："
        return prompt

    def parse_output(self, raw_text):
        """
        极其强壮的容错解析器：即使模型没写闭合标签，也能强行拆分
        """
        analysis = "（模型未生成显式分析）"
        reply = raw_text

        # 1. 尝试用 </analysis> 作为物理分割点
        if "</analysis>" in raw_text:
            parts = raw_text.split("</analysis>")
            # 上半部分是分析
            analysis = parts[0].strip() + "\n</analysis>"
            # 下半部分是回复
            reply_raw = parts[1]

            # 进一步剥离 <response> 标签
            if "<response>" in reply_raw:
                reply = reply_raw.split("<response>")[-1]
            else:
                reply = reply_raw

            reply = reply.replace("</response>", "").strip()

        else:
            # 2. 极端容错：如果连 </analysis> 都没有，说明全都是分析（被截断了）
            # 或者模型根本没按常理出牌
            if "<response>" in raw_text:
                parts = raw_text.split("<response>")
                analysis = parts[0].strip()
                reply = parts[1].replace("</response>", "").strip()
            else:
                analysis = raw_text
                reply = "（回复被截断或生成异常，请引导我继续：比如输入'继续'）"

        return analysis, reply

    def chat(self, user_input):
        # 1. 构造基础 Prompt
        prompt = self.build_prompt(user_input)

        # 🔥 核心绝招：前缀引导 (Prefilling)
        # 我们不仅让它“请回复：”，更是直接替它写下第一句分析的开头！
        # 这样模型就别无选择，只能顺着“【情绪诊断】”往下写，瞬间进入潜台词模式。
        forced_prefix = "<analysis>\n【情绪诊断】"
        prompt += forced_prefix

        # 🌟 关键修复：保持与训练时 Tokenizer 一致！
        # 你在 train_chair_final.py 中训练时用了 add_special_tokens=False
        # 如果推理时不加这个，ChatGLM2 会自动塞入 [gMASK] 等特殊符，导致模型发懵
        inputs = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # 限制单次生成长度，防止废话
                top_p=0.8,  # 稍微收敛一点发散性
                temperature=0.4,  # 降低温度，让模型严格按照我们训练的思维链格式输出
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 3. 截取新生成的部分（剥离掉输入的 prompt）
        input_length = inputs["input_ids"].shape[1]
        response_ids = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # 4. 拼回我们强制加的前缀，保证 parse_output 收到的是完整结构
        full_text = forced_prefix + generated_text

        # 5. 鲁棒解析
        analysis, reply = self.parse_output(full_text)

        # 6. 更新历史 (注意：只存纯净的 reply，绝不能把 analysis 存进历史！)
        if reply != "（回复被截断或生成异常，请引导我继续：比如输入'继续'）":
            self.history.append(("client", user_input))
            self.history.append(("counselor", reply))

        return analysis, reply


# ================= 主运行循环 =================
def main():
    agent = ChairInference()
    print("\n" + "=" * 50)
    print("💡 心理咨询室已开启 (输入 'exit' 或 'quit' 退出)")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n👤 来访者: ").strip()
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]:
                print("👋 咨询结束，祝您生活愉快。")
                break

            # 获取回复
            analysis, reply = agent.chat(user_input)

            # 打印“大脑思考过程” (灰色显示，模拟后台数据)
            print("\n" + "-" * 30)
            print(f"\033[90m🧠 [思维链/潜台词]\n{analysis}\033[0m")
            print("-" * 30)

            # 打印最终回复
            print(f"\n👩‍⚕️ [主席智能体]:\n{reply}")

        except KeyboardInterrupt:
            print("\n👋 用户中断。")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()