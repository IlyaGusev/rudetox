import argparse
from collections import Counter

import razdel

from util import read_jsonl, write_jsonl


def main(
    input_path,
    output_path,
    bad_vocab_path
):
    vocab = set()
    with open(bad_vocab_path) as r:
        for line in r:
            vocab.add(line.strip())

    records = read_jsonl(input_path)
    filtered_records = []
    non_toxic_bad_words_count = 0
    for record in records:
        label = record["label"]
        text = record["text"]
        tokens = [t.text.lower() for t in razdel.tokenize(text)]
        has_bad_token = False
        for token in tokens:
            if token in vocab:
                has_bad_token = True
        if has_bad_token and label != 1:
            non_toxic_bad_words_count += 1
            continue
        filtered_records.append(record)
    print("Non toxic with bad words:", non_toxic_bad_words_count)
    write_jsonl(filtered_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--bad-vocab-path", required=True, type=str)
    args = parser.parse_args()
    main(**vars(args))
