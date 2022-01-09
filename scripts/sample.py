import argparse
import sys
import re
import string

from util.io import read_jsonl, write_jsonl
from util.dl import Classifier


def is_clean(text):
    remainder = re.sub(r'[\w\s{}]+'.format(string.punctuation), '', text).strip()
    if remainder:
        return False

    punct_count = 0
    for ch in text:
        if ch in string.punctuation:
            punct_count += 1
    if punct_count >= 5:
        return False

    has_latin = any("a" <= ch <= "z" for ch in text)
    if has_latin:
        return False

    if "#" in text:
        return False

    numbers_count = 0
    for ch in text:
        if ch.isnumeric():
            numbers_count += 1
    if numbers_count >= 3:
        return False

    return True


def main(
    input_path,
    output_path,
    good_label,
    exclude_sources,
    fluent_only,
    sample_rate
):
    exclude_sources = exclude_sources.split(",")
    records = read_jsonl(input_path, sample_rate)

    texts = []
    sources = dict()
    for record in records:
        text = record["text"]
        label = record["label"]
        source = record["source"]
        sources[text] = source
        if label != good_label:
            continue
        if source in exclude_sources:
            continue
        if not (25 < len(text) < 150) or not is_clean(text):
            continue
        texts.append(text)


    if fluent_only:
        fluency_model = Classifier("SkolkovoInstitute/rubert-base-corruption-detector")
        labels = fluency_model(texts)
        filtered_texts = []
        for label, text in zip(labels, texts):
            if label == 1:
                filtered_texts.append(text)
        texts = filtered_texts

    write_jsonl([{"text": t, "source": sources[t]} for t in texts], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--good-label", type=int, default=1)
    parser.add_argument("--exclude-sources", type=str, default="detox")
    parser.add_argument("--fluent-only", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
