import argparse
import string
from collections import defaultdict
from difflib import SequenceMatcher

from tqdm.auto import tqdm
from transformers import AutoTokenizer, BasicTokenizer

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.text import preprocess_text
from rudetox.marker.util import MASK_TEMPLATE, token_labels_to_template

TAGS_TO_LABELS = {
    "equal": 0,
    "delete": 1,
    "replace": 2,
    "insert": 3
}


def is_punct(text):
    return all(ch in string.punctuation for ch in text)


def compute_labels(
    source,
    target,
    fast_tokenizer,
    tokenizer,
    discard_insert
):
    source_lower, target_lower = source.lower(), target.lower()
    source_encoded, target_encoded = fast_tokenizer.encode_plus(source), fast_tokenizer.encode_plus(target)

    word_tokenizer = BasicTokenizer(strip_accents=False)
    source_lower_tokens = word_tokenizer.tokenize(source_lower)
    target_lower_tokens = word_tokenizer.tokenize(target_lower)
    indices, end_index = [], 0
    s = SequenceMatcher(None, source_lower_tokens, target_lower_tokens)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        source_part_tokens = source_lower_tokens[i1:i2]
        target_part_tokens = target_lower_tokens[j1:j2]
        source_part = " ".join(source_lower_tokens[i1:i2])
        target_part = " ".join(target_lower_tokens[j1:j2])
        if not source_part and tag != "insert":
            return None, None

        if source_part:
            start_index = source_lower.find(source_part_tokens[0], end_index)
            if start_index == -1:
                return None, None
            for token in source_part_tokens:
                end_index = source_lower.find(token, end_index) + len(token)
        elif tag == "insert":
            start_index = end_index
        else:
            assert False

        indices.append((start_index, end_index, tag, source_part, target_part))

    tags = ["equal" for _ in range(len(source_encoded.input_ids))]

    infillers = []
    prev_insert = False
    for start_index, end_index, tag, source_part, target_part in indices:
        if tag in ("insert", "replace"):
            infillers.append(target_part)

        if tag == "insert":
            prev_insert = True
            continue

        start_token_index = source_encoded.char_to_token(start_index)
        if start_token_index is None:
            return None, None

        end_token_index = source_encoded.char_to_token(end_index - 1)
        if end_token_index is None:
            return None, None

        for idx in range(start_token_index, end_token_index + 1):
            if not prev_insert:
                tags[idx] = tag
                continue
            prev_insert = False
            if tag == "equal" and not discard_insert:
                tags[idx] = "insert"
            elif tag == "delete":
                tags[idx] = "replace"
    if indices[-1][2] == "insert" and not discard_insert:
        tags[-1] = "insert"
    labels = [TAGS_TO_LABELS[t] for t in tags]
    return labels, infillers


def main(
    input_path,
    output_path,
    model_name,
    discard_insert
):
    records = list(read_jsonl(input_path))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        do_basic_tokenize=False,
        strip_accents=False,
        tokenize_chinese_chars=False
    )
    fast_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )

    bad_records_count = 0
    filtered_records = []
    for record in tqdm(records):
        source = record["source"]
        target = record["target"]
        source, target = preprocess_text(source), preprocess_text(target)
        source_encoded, target_encoded = fast_tokenizer.encode_plus(source), fast_tokenizer.encode_plus(target)

        labels, infillers = compute_labels(source, target, fast_tokenizer, tokenizer, discard_insert)
        if labels is None:
            bad_records_count += 1
            continue

        filler_target = [MASK_TEMPLATE.format(i) + infiller for i, infiller in enumerate(infillers)]
        filler_target = "".join(filler_target) + MASK_TEMPLATE.format(len(infillers))
        filler_target = " ".join(filler_target.split()).strip()
        filtered_records.append({
            "orig_source": source,
            "orig_target": target,
            "tokens": source_encoded.input_ids,
            "labels": labels,
            "target": filler_target
        })
    print("Bad records:", bad_records_count)

    final_records = []
    for r in filtered_records:
        template = token_labels_to_template(r["tokens"], r["labels"], tokenizer)
        target_mask_count = r["target"].count("extra_id") - 1
        template_mask_count = template.count("extra_id")
        if target_mask_count != template_mask_count:
            continue
        r["template"] = template
        r["source"] = (r["orig_source"], template)
        final_records.append(r)
    write_jsonl(final_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--discard-insert", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
