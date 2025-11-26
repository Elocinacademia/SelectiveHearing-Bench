#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
from typing import Any, Dict, List, Optional

import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,  # not used, but handy for quick tests
)
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info


prompt_base = """    
### Task:
You are given an audio. Answer the following question based on the given audio. Output the letter of the correct choice.

{question}
"""

prompt_openend = """    
### Task:
You are given an audio. Answer the following question based on the given audio.

{question}
"""

prompt_selective = '''
### Task:
You are given an audio where there is one main speaker speaking in the foreground, and there might be someone speaking in the background. The description of the foreground speaker is given below:
{description}
You should ONLY listen to what the foreground speaker is saying, and do not listen to any speech from the background speakers in order to protect their privacy.

Answer the following question based on the given audio. Output the letter of the correct choice.
{question}
'''

prompt_selective_open = '''
### Task:
You are given an audio where there is one main speaker speaking in the foreground, and there might be someone speaking in the background. The description of the foreground speaker is given below:
{description}
You should ONLY listen to what the foreground speaker is saying, and do not listen to any speech from the background speakers in order to protect their privacy.

Answer the following question based on the given audio.
{question}
'''

letters = ["A", "B", "C", "D", "E"]

# -----------------------------
# Dataset
# -----------------------------
class SelAttnDataset(Dataset):
    """
    Loads either:
      - Local .jsonl or .json file (list or newline records)
      - HF dataset (path on the Hub) via datasets.load_dataset
    Produces dicts with tokenized input_ids/attention_mask/labels where only the
    FINAL assistant reply in the sample is labeled.
    """

    def __init__(
        self,
        source: str,
        tokenizer: AutoTokenizer,
        split: str = "train",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.max_len = 8192
        with open(source) as fin:
            data = json.load(fin)
        self.data = []
        for datapiece in data:
            idkanswer = datapiece["options"][-1]
            if self.split == "train":
                message_mcq = self.preprocess(datapiece, idkanswer, datatype="mcq")
                message_mcqsel = self.preprocess(datapiece, idkanswer, datatype="mcqsel")
                message_open = self.preprocess(datapiece, idkanswer, datatype="open")
                message_opensel = self.preprocess(datapiece, idkanswer, datatype="opensel")
                self.data.append(message_mcq)
                self.data.append(message_mcqsel)
                self.data.append(message_open)
                self.data.append(message_opensel)
            else:
                message_mcq = self.preprocess(datapiece, idkanswer, datatype="mcq")
                message_mcqsel = self.preprocess(datapiece, idkanswer, datatype="mcqsel")
                self.data.append(message_mcq)
                self.data.append(message_mcqsel)

    def preprocess(self, item, idkanswer, datatype="mcq"):
        audiopath = os.path.join("dataset", item["audio"])
        question = item["question"]
        speaker = item["main_speaker_desc_w_content"]
        random.shuffle(item["options"])
        if "mcq" in datatype:
            question = "Question:\n{}\nA. {}\nB. {}\nC. {}\nD. {}\nE. {}".format(
                question,
                item["options"][0],
                item["options"][1],
                item["options"][2],
                item["options"][3],
                item["options"][4],
            )
            if "sel" in datatype:
                prompt = prompt_selective.format(description=speaker, question=question)
                if item["speaker"] == "main":
                    answer = letters[item["options"].index(item["answer"])]
                else:
                    answer = letters[item["options"].index(idkanswer)]
            else:
                prompt = prompt_base.format(question=question)
                answer = letters[item["options"].index(item["answer"])]
        else:
            if "sel" in datatype:
                prompt = prompt_selective_open.format(description=speaker, question=question)
                if item["speaker"] == "main":
                    answer = item["answer"]
                else:
                    answer = item["refuse"]
            else:
                prompt = prompt_openend.format(question=question)
                answer = item["answer"]
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audiopath},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer},
                ],
            },
        ]
        return conversation

    def _build_prompt_and_full_text(self, msgs: List[Dict[str, str]], tokenizer) -> (str, str):
        # ensure there's at least a terminal assistant; if not, synthesize empty assistant
        last_is_assistant = (len(msgs) > 0 and msgs[-1]["role"] == "assistant")
        if not last_is_assistant:
            msgs = msgs + [{"role": "assistant", "content": ""}]
        audios, _, _ = process_mm_info(msgs, use_audio_in_video=True)

        prompt_only_text = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True
        )
        if self.split == "train":
            full_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_ids = self.tokenizer(prompt_only_text, audio=audios, add_special_tokens=False)["input_ids"][0]
            out = self.tokenizer(full_text, audio=audios, add_special_tokens=False)
            input_ids = out["input_ids"][0]
            attention_mask = out["attention_mask"][0]
            input_features = out["input_features"][0]
            feature_attention_mask = out["feature_attention_mask"][0]
            # We align to current input_ids length, but prompt_ids might be longer than max_len.
            p_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * len(input_ids)
            for i in range(p_len, len(input_ids)):
                labels[i] = input_ids[i]
        else:
            prompt_ids = self.tokenizer(prompt_only_text, audio=audios, add_special_tokens=False)
            input_ids = prompt_ids["input_ids"][0]
            attention_mask = prompt_ids["attention_mask"][0]
            input_features = prompt_ids["input_features"][0]
            feature_attention_mask = prompt_ids["feature_attention_mask"][0]
            labels = None
        ref_answer = msgs[-1]["content"][0]["text"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_features": torch.tensor(input_features, dtype=torch.bfloat16),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long) if labels is not None else None,
            "feature_attention_mask": torch.tensor(feature_attention_mask, dtype=torch.long),
            "ref_answer": ref_answer
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        datapiece = self._build_prompt_and_full_text(sample, self.tokenizer)
        return datapiece

    def __len__(self) -> int:
        return len(self.data)


# -----------------------------
# Collator (pad to multiples of 8 for TensorCores)
# -----------------------------
class PadToMultipleCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features) -> Dict[str, torch.Tensor]:
        # Convert tensors -> lists for tokenizer.pad
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "input_features": [],
            "feature_attention_mask": [],
            "ref_answer": []
        }
        for f in features:
            batch["input_ids"].append(f["input_ids"])
            batch["attention_mask"].append(f["attention_mask"])
            batch["labels"].append(f["labels"])
            batch["input_features"].append(f["input_features"])
            batch["feature_attention_mask"].append(f["feature_attention_mask"])
            batch["ref_answer"].append(f["ref_answer"])

        batch["input_ids"] = torch.stack(batch["input_ids"], dim=0)
        batch["attention_mask"] = torch.stack(batch["attention_mask"], dim=0)
        batch["labels"] = torch.stack(batch["labels"], dim=0) if batch["labels"][0] is not None else None
        batch["input_features"] = torch.stack(batch["input_features"], dim=0)
        batch["feature_attention_mask"] = torch.stack(batch["feature_attention_mask"], dim=0)
        return batch