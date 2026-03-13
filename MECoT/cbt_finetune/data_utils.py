# data_utils.py
import json
import glob
import torch
from torch.utils.data import Dataset
import os
from typing import List, Dict, Tuple

# 定义 CBT 系统提示词
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


class CBTConversationDataset(Dataset):
    def __init__(self, data_args, tokenizer):
        self.data_args = data_args
        self.tokenizer = tokenizer
        self.examples = self._load_and_process(data_args.data_dir)

    def _load_and_process(self, data_dir):
        """
        读取目录下所有json，并将其扁平化为 (query, response, history) 的形式
        """
        file_paths = glob.glob(os.path.join(data_dir, "*.json"))
        examples = []

        print(f"Loading data from {len(file_paths)} files...")

        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    session = json.load(f)

                # 维护当前 session 的历史
                history = []

                for i, turn in enumerate(session):
                    # 我们只训练模型充当 counselor 的时刻
                    if turn['role'] == 'counselor':
                        response = turn['content']

                        # 找到对应的上一句 client
                        if i > 0 and session[i - 1]['role'] == 'client':
                            query = session[i - 1]['content']

                            examples.append({
                                "query": query,
                                "response": response,
                                "history": history.copy()
                            })

                            # 更新历史，供下一轮使用
                            history.append((query, response))

            except Exception as e:
                print(f"Skipping {path}: {e}")

        print(f"Loaded {len(examples)} training samples.")
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CBTDataCollator:
    def __init__(self, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __call__(self, batch: List[Dict]):
        # ChatGLM2 需要手动拼接 prompt
        input_ids_batch = []
        labels_batch = []

        for instance in batch:
            query = instance['query']
            response = instance['response']
            history = instance['history']

            # 1. 构建 Prompt (Source)
            # 格式参考 ChatGLM2 官方: [Round 1]\n\n问：...\n\n答：...
            prompt = ""
            if len(history) == 0:
                # 第一轮加入 System Prompt
                prompt += CBT_SYSTEM_PROMPT + "\n\n"

            for i, (old_q, old_a) in enumerate(history):
                prompt += f"[Round {i + 1}]\n\n问：{old_q}\n\n答：{old_a}\n\n"

            prompt += f"[Round {len(history) + 1}]\n\n问：{query}\n\n答："

            # 2. 编码
            # add_special_tokens=False 是因为我们手动控制 EOS
            source_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            target_ids = self.tokenizer.encode(response, add_special_tokens=False)
            eos_id = self.tokenizer.eos_token_id

            # 3. 截断 (从左侧截断历史，保留最新的 context)
            if len(source_ids) > self.max_source_len:
                source_ids = source_ids[-self.max_source_len:]

            if len(target_ids) > self.max_target_len:
                target_ids = target_ids[:self.max_target_len]

            # 4. 拼接 input_ids
            # input = prompt + response + eos
            input_ids = source_ids + target_ids + [eos_id]

            # 5. 构建 labels (Loss Masking)
            # prompt 部分设为 -100， response 部分保留
            labels = [-100] * len(source_ids) + target_ids + [eos_id]

            input_ids_batch.append(torch.tensor(input_ids, dtype=torch.long))
            labels_batch.append(torch.tensor(labels, dtype=torch.long))

        # 6. Padding
        # 使用 RNN pad sequence 进行动态 padding
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