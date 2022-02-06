import argparse
import json
from collections import Counter

from rudetox.util.io import read_jsonl, write_jsonl


def is_good_record(r, config):
    scores = r["scores"]
    target = r["target"]
    source = r["source"]
    ban_words = config.get("ban_words", [])
    if scores["style"] > config.get("max_style", 1.0):
        return False, "max_style"
    if scores["style"] < config.get("min_style", 0.0):
        return False, "min_style"
    if scores["fluency"] < config.get("min_fluency", 0.0):
        return False, "fluency"
    if scores["sim"] < config.get("min_sim", 0.0):
        return False, "sim"
    if scores["chrf"] < config.get("min_chrf", 0.0):
        return False, "chrf"
    return True, "ok"


def main(
    input_path,
    output_path,
    config_path,
    source_field,
    target_field
):
    records = list(read_jsonl(input_path))
    for r in records:
        source = r.pop(source_field)
        target = r.pop(target_field)
        r["source"] = source
        r["target"] = target

    skip_reasons = Counter()
    with open(config_path) as r:
        config = json.load(r)

    filtered_records = []
    for r in records:
        flag, reason = is_good_record(r, config)
        if not flag:
            skip_reasons[reason] += 1
            continue
        filtered_records.append(r)
    print(skip_reasons.most_common())
    write_jsonl(filtered_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--source-field", type=str, required=True)
    parser.add_argument("--target-field", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
