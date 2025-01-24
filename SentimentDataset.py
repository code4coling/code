#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import torch, json, copy
 
PROMPTS = {
    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Response:"
}

class SentimentInstructionDataset(Dataset):
    def __init__(self, path, tokenizer, split="train", max_words=224):
        self.data = json.load(open(path))
        self.data = self.data[split]
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        data = self.data[index]
        if data.get("input", "") == "":
            prompt = PROMPTS["prompt_no_input"].format_map(data)
        else:
            prompt = PROMPTS["prompt_input"].format_map(data)
        example = prompt + data["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }