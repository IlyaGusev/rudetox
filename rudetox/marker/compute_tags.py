import sys
from difflib import SequenceMatcher

from transformers import AutoTokenizer

from util import read_jsonl, write_jsonl

input_path = sys.argv[1]
model_name = sys.argv[2]
output_path = sys.argv[3]

filtered_records = dict()
records = list(read_jsonl(input_path))
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
for record in records:
    source, target = record["source"], record["target"]
    source_lower, target_lower = source.lower(), target.lower()

    source_encoded = tokenizer.encode_plus(source)
    target_encoded = tokenizer.encode_plus(target)
    source_lower_ids = tokenizer(source_lower, add_special_tokens=False)["input_ids"]
    target_lower_ids = tokenizer(target_lower, add_special_tokens=False)["input_ids"]

    s = SequenceMatcher(None, source_lower_ids, target_lower_ids)
    indices = []
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        source_part = tokenizer.decode(source_lower_ids[i1:i2]).replace("<unk>", "").strip(", .?!")
        target_part = tokenizer.decode(target_lower_ids[j1:j2])
        if len(source_part) <= 2:
            continue
        if tag in ("equal", "insert"):
            continue
        start_index = source_lower.find(source_part)
        end_index = start_index + len(source_part)
        if start_index == -1:
            source_part = source_part.replace(",", " ,")
            start_index = source_lower.find(source_part)
            end_index = start_index + len(source_part)
        if start_index == -1:
            continue
        indices.append((start_index, end_index))

    tags = [0 for _ in range(len(source_encoded["input_ids"]))]
    target_tags = [0 for _ in range(len(target_encoded["input_ids"]))]
    for start_index, end_index in indices:
        start_token_index = source_encoded.char_to_token(start_index)
        end_token_index = source_encoded.char_to_token(end_index - 1) + 1
        for idx in range(start_token_index, end_token_index):
            tags[idx] = 1
    filtered_records[source] = {
        "text": source,
        "labels": tags
    }
    filtered_records[target] = {
        "text": target,
        "labels": target_tags
    }

write_jsonl(list(filtered_records.values()), output_path)
