import argparse
from difflib import SequenceMatcher

from transformers import AutoTokenizer

from rudetox.util.io import read_jsonl, write_jsonl


def main(
    input_path,
    output_path,
    model_name
):
    filtered_records = dict()
    records = list(read_jsonl(input_path))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    for record in records:
        source, target = record["source"], record["target"]
        source_lower, target_lower = source.lower(), target.lower()
        source_encoded, target_encoded = tokenizer.encode_plus(source), tokenizer.encode_plus(target)

        source_lower_ids = tokenizer(source_lower, add_special_tokens=False).input_ids
        target_lower_ids = tokenizer(target_lower, add_special_tokens=False).input_ids

        s = SequenceMatcher(None, source_lower_ids, target_lower_ids)
        indices = []
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            source_part = tokenizer.decode(source_lower_ids[i1:i2])
            source_part = source_part.replace(tokenizer.unk_token, "").strip(", .?!|:;#-d)(")
            target_part = tokenizer.decode(target_lower_ids[j1:j2])
            if len(source_part) <= 2:
                continue
            if tag in ("equal", "insert"):
                continue
            start_index = source_lower.find(source_part)

            replacements = {
                ",": " ,",
                " - ": "-",
                " -": "-",
                " )": ")",
                ". ": ".",
                " , ": ",",
                "! ": "!"
            }
            for rp_from, rp_to in replacements.items():
                if start_index != -1:
                    break
                source_part = source_part.replace(rp_from, rp_to)
                start_index = source_lower.find(source_part)

            if start_index == -1:
                continue
            end_index = start_index + len(source_part)
            indices.append((start_index, end_index))

        tags = [0 for _ in range(len(source_encoded.input_ids))]
        target_tags = [0 for _ in range(len(target_encoded.input_ids))]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
