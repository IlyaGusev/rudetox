import random
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class Seq2seqBaseDataset(Dataset):
    def __init__(
        self,
        original_records: List[Dict],
        sample_rate: float,
        tokenizer: AutoTokenizer,
        max_source_tokens_count: int,
        max_target_tokens_count: int,
        source_field: str = "source",
        target_field: str = "target",
        style_field: str = None
    ):
        self.original_records = original_records
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_source_tokens_count = max_source_tokens_count
        self.max_target_tokens_count = max_target_tokens_count
        self.source_field = source_field
        self.target_field = target_field
        self.style_field = style_field

        self.records = []
        for record in tqdm(original_records):
            if random.random() > self.sample_rate:
                continue
            tensors = self.convert_pair(
                source=record[source_field],
                target=record.get(target_field),
                style=record.get(style_field) if style_field else None
            )
            self.records.append(tensors)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def convert_pair(self, source, target, style):
        raise NotImplementedError


class Seq2seqDataset(Seq2seqBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_pair(self, source, target, style):
        if style is not None:
            style_token = "<extra_id_1>" if style == 1 else "<extra_id_0>"
            source = style_token + " " + source
        inputs = self.tokenizer(
            source,
            add_special_tokens=True,
            max_length=self.max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        if target is not None:
            outputs = self.tokenizer(
                target,
                add_special_tokens=True,
                max_length=self.max_target_tokens_count,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            labels = outputs["input_ids"].squeeze(0)
            labels[outputs["attention_mask"].squeeze(0) == 0] = -100
            inputs["labels"] = labels
        return inputs
