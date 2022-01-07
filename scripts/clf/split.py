import argparse
from collections import Counter, defaultdict
import random

from util.io import read_jsonl, write_jsonl


def main(
    input_path,
    train_path,
    val_path,
    test_path,
    val_border,
    test_border,
    seed
):
    random.seed(seed)
    records = list(read_jsonl(input_path))
    random.shuffle(records)

    source_records = defaultdict(list)
    train_records, val_records, test_records = [], [], []
    for record in records:
        source = record["source"]
        source_records[source].append(record)

    for source, r in source_records.items():
        n = len(r)
        train_records.extend(r[:int(n * val_border)])
        val_records.extend(r[int(n * val_border): int(n * test_border)])
        test_records.extend(r[int(n * test_border):])

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)
    write_jsonl(test_records, test_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--train-path", required=True, type=str)
    parser.add_argument("--val-path", required=True, type=str)
    parser.add_argument("--test-path", required=True, type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--val-border", default=0.8, type=float)
    parser.add_argument("--test-border", default=0.9, type=float)
    args = parser.parse_args()
    main(**vars(args))
