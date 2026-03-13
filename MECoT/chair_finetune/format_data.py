import json
import os
import random

# ==========================================
# 1. 训练专用 Prompt 设计 (基于构造代码改良)
# ==========================================
# 这里的 Prompt 去掉了“输入数据”中的专家建议部分，
# 而是改为要求模型“自行生成”这些分析作为思维链。
TRAIN_SYSTEM_PROMPT = """你不是普通的心理咨询师，你是**"多专家会诊系统"的首席决策者（主席智能体）**。
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


def format_chair_data(input_dir, output_file):
    formatted_data = []

    # 遍历你的数据目录
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"找到 {len(files)} 个数据文件，开始构建思维链数据...")

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
                    # 获取元数据，这是构建大脑分析的关键！
                    meta = target_turn.get('metadata', {})
                    emo = meta.get('emotion_analysis', {})
                    cbt_suggestion = meta.get('cbt_expert_response', '（无CBT建议）')
                    human_suggestion = meta.get('humanistic_expert_response', '（无人本建议）')

                    # ==========================================
                    # 2. 构建思维链目标 (Target Construction)
                    # ==========================================
                    # 我们把原本作为Input的专家建议，变成模型需要学习输出的Internal Thought
                    cot_content = (
                        f"<analysis>\n"
                        f"【情绪诊断】核心情绪：{emo.get('core_emotion', '未知')} (强度: {emo.get('intensity', 5)}/10)；压力源：{emo.get('key_stressor', '未知')}\n"
                        f"【CBT视角推演】{cbt_suggestion[:200]}...\n"  # 截取精华，避免过长
                        f"【人本视角推演】{human_suggestion[:200]}...\n"
                        f"</analysis>"
                    )

                    # 最终模型的训练目标 = 思考过程 + 最终回复
                    final_output = f"{cot_content}\n\n<response>\n{target_turn['content']}\n</response>"

                    # ==========================================
                    # 3. 构建历史上下文 (Context)
                    # ==========================================
                    input_text = ""
                    if history:
                        input_text += "【对话历史摘要】：\n"
                        # 只取最近 3 轮，防止上下文过长导致幻觉
                        for h_role, h_content in history[-6:]:
                            tag = "来访者" if h_role == "client" else "咨询师"
                            input_text += f"{tag}：{h_content}\n"

                    input_text += f"\n【当前来访者诉求】：{user_content}"

                    # 构造 ChatGLM2 格式
                    formatted_data.append({
                        "instruction": TRAIN_SYSTEM_PROMPT,
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

    print(f"✅ 数据转换完成！共生成 {len(formatted_data)} 条样本。")
    print(f"📂 输出文件: {output_file}")


if __name__ == "__main__":
    # 请确保这里的目录指向你存放 chair.json 的位置
    format_chair_data("../data_chair_enhance_cleaned", "train_data_final.json")