# data_utils.py
import json
import glob
import torch
from torch.utils.data import Dataset
import os
from typing import List, Dict

# ==============================================================================
# 【核心修改】人本主义 (Humanistic) 专用 System Prompt
# ==============================================================================
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


# ==============================================================================

class HumanisticDataset(Dataset):
    def __init__(self, data_args, tokenizer):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.examples = self._load_and_process(data_args.data_dir)

    def _load_and_process(self, data_dir):
        file_paths = glob.glob(os.path.join(data_dir, "*.json"))
        examples = []

        print(f"正在加载人本主义数据: {len(file_paths)} 个文件...")

        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    session = json.load(f)

                history = []
                for i, turn in enumerate(session):
                    if turn['role'] == 'counselor':
                        response = turn['content']
                        if i > 0 and session[i - 1]['role'] == 'client':
                            query = session[i - 1]['content']

                            examples.append({
                                "query": query,
                                "response": response,
                                "history": history.copy()
                            })
                            history.append((query, response))
            except Exception as e:
                print(f"Error reading {path}: {e}")

        print(f"加载完成，共生成 {len(examples)} 条训练样本。")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class HumanisticCollator:
    def __init__(self, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict]):
        input_ids_batch = []
        labels_batch = []

        for instance in batch:
            query = instance['query']
            response = instance['response']
            history = instance['history']

            # 构建 Prompt
            prompt = ""
            if len(history) == 0:
                # 第一轮对话注入人本主义 System Prompt
                prompt += HUMANISTIC_SYSTEM_PROMPT + "\n\n"

            for i, (q, a) in enumerate(history):
                prompt += f"[Round {i + 1}]\n\n问：{q}\n\n答：{a}\n\n"

            prompt += f"[Round {len(history) + 1}]\n\n问：{query}\n\n答："

            # 编码与截断
            source_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(response, add_special_tokens=False)
            eos_id = self.tokenizer.eos_token_id

            if len(source_ids) > self.max_source_len:
                source_ids = source_ids[-self.max_source_len:]
            if len(target_ids) > self.max_target_len:
                target_ids = target_ids[:self.max_target_len]

            input_ids = source_ids + target_ids + [eos_id]
            # Loss Masking: 仅计算咨询师回答的 Loss
            labels = [-100] * len(source_ids) + target_ids + [eos_id]

            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))

        # Padding
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            input_ids_batch, batch_first=True,
            padding_value=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else 0
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            labels_batch, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids_padded,
            "labels": labels_padded
        }