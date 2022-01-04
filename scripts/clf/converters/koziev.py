import argparse
import string
import re

from tqdm import tqdm

from util import write_jsonl


def remove_punct(s):
    s = re.sub(r'[^\w\s]', '', s)
    s = " ".join(s.split())
    return s


def main(
    input_file,
    output_file,
    vocab_file
):
    bad_words = set()
    with open(vocab_file) as r:
        for line in r:
            line = line.strip()
            bad_words.add(line)

    records = dict()
    with open(input_file) as r:
        for line in tqdm(r):
            line = line.strip()
            if len(line) < 5:
                continue
            line = line[2:]
            tokens = remove_punct(line).split()
            has_bad_words = any(token in bad_words for token in tokens)
            if has_bad_words:
                records[line] = {
                    "text": line,
                    "label": 1,
                    "source": "koziev"
                }
    write_jsonl(list(records.values()), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--vocab-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
