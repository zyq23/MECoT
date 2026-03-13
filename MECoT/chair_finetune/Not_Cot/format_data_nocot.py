import json
import os
import random

# ==========================================
# 1. No-CoT 专用 System Prompt
# ==========================================
# 相比原版，去掉了“第一步：深度分析”的要求，
# 保留了对“主席智能体”身份和“融合流派”风格的定义。
TRAIN_SYSTEM_PROMPT_NOCOT = """你不是普通的心理咨询师，你是**"多专家会诊系统"的首席决策者（主席智能体）**。
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


def format_chair_data_nocot(input_dir, output_file):
    formatted_data = []

    # 遍历你的数据目录
    if not os.path.exists(input_dir):
        print(f"❌ 错误：输入目录 {input_dir} 不存在！")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"找到 {len(files)} 个数据文件，开始构建 No-CoT (Input->Response) 数据...")

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dialog = json.load(f)
        except Exception as e:
            print(f"读取文件 {filename} 失败: {e}")
            continue

        history = []

        # 遍历对话
        for i, turn in enumerate(dialog):
            if turn['role'] == 'client':
                user_content = turn['content']

                # 寻找下一轮咨询师（主席）的回复
                if i + 1 < len(dialog) and dialog[i + 1]['role'] == 'counselor':
                    target_turn = dialog[i + 1]

                    # ==========================================
                    # 2. 构建目标 (Target Construction) - 核心修改
                    # ==========================================
                    # 这是一个标准的 SFT 目标，只有最终回复，没有 <analysis>
                    # 为了保持一致性，我们这里甚至可以不加 <response> 标签，直接输出文本
                    # 或者为了解析方便保留 <response>，这里选择直接输出文本以模拟纯粹的端到端对话
                    final_output = target_turn['content']

                    # ==========================================
                    # 3. 构建历史上下文 (Context)
                    # ==========================================
                    input_text = ""
                    if history:
                        input_text += "【对话历史摘要】：\n"
                        # 只取最近 6 轮
                        for h_role, h_content in history[-6:]:
                            tag = "来访者" if h_role == "client" else "咨询师"
                            input_text += f"{tag}：{h_content}\n"

                    input_text += f"\n【当前来访者诉求】：{user_content}"

                    # 构造 ChatGLM2 格式
                    formatted_data.append({
                        "instruction": TRAIN_SYSTEM_PROMPT_NOCOT,
                        "input": input_text,
                        "output": final_output
                    })

                    # 更新历史
                    history.append(("client", user_content))
                    history.append(("counselor", target_turn['content']))

    # 打乱数据
    random.shuffle(formatted_data)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, ensure_ascii=False, indent=2)

    print(f"✅ No-CoT 数据转换完成！共生成 {len(formatted_data)} 条样本。")
    print(f"📂 输出文件: {output_file}")


if __name__ == "__main__":
    # 指向你的原始数据目录
    format_chair_data_nocot("/data/zyq/mental_health_debate_framework/data_chair_enhance_cleaned", "train_data_nocot.json")