import argparse
import random
import json
import copy
from collections import Counter

from rudetox.util.io import read_jsonl, write_jsonl, read_lines
from rudetox.clf.transformations import form_transformations


def main(
    input_path,
    output_path,
    config_path,
    text_field,
    label_field,
    seed,
    bad_vocab_path
):
    random.seed(seed)
    with open(config_path) as r:
        config = json.load(r)
    augmentations = config["augmentations"]

    records = list(read_jsonl(input_path))
    toxic_texts = [r[text_field] for r in records if r[label_field] == 1]
    non_toxic_texts = [r[text_field] for r in records if r[label_field] == 0]
    toxic_words = read_lines(bad_vocab_path)
    transformations = form_transformations(toxic_texts, non_toxic_texts, toxic_words)

    counts = Counter()
    augmented_records = []
    for record in records:
        for aug in augmentations:
            aug_name = aug["name"]
            augmented_text = transformations[aug_name](record[text_field])
            if not augmented_text:
                continue

            aug_rate = float(aug.get("rate", 1.0))
            if random.random() > aug_rate:
                continue

            counts[aug_name] += 1
            new_record = copy.deepcopy(record)
            new_record[text_field] = augmented_text
            new_record["aug"] = aug_name

            if "label" in aug:
                label = int(aug["label"])
                new_record["label"] = label
            augmented_records.append(new_record)

    for name, cnt in counts.most_common():
        print("{}: {}".format(name, cnt))
    records.extend(augmented_records)
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--bad-vocab-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--label-field", type=str, default="label")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))
