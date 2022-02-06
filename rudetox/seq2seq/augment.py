import argparse
import random
import json
import copy
from collections import Counter

from tqdm import tqdm

from rudetox.util.io import read_jsonl, write_jsonl, read_lines
from rudetox.transformations import form_transformations


def main(
    input_path,
    output_path,
    config_path,
    source_field,
    bad_vocab_path,
    seed
):
    random.seed(seed)
    with open(config_path) as r:
        config = json.load(r)
    augmentations = config["augmentations"]

    records = list(read_jsonl(input_path))
    toxic_words = read_lines(bad_vocab_path)
    transformations = form_transformations(toxic_words=toxic_words)

    counts = Counter()
    augmented_records = []
    for record in tqdm(records):
        for aug in augmentations:
            aug_name = aug["name"]
            aug_rate = float(aug.get("rate", 1.0))
            if random.random() > aug_rate:
                continue

            augmented_source = transformations[aug_name](record[source_field])
            if not augmented_source:
                continue

            counts[aug_name] += 1
            new_record = copy.deepcopy(record)
            new_record[source_field] = augmented_source
            new_record["aug"] = aug_name
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
    parser.add_argument("--source-field", type=str, default="source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(**vars(args))
