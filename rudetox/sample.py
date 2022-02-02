import argparse
import json

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.text import preprocess_text


def main(
    input_path,
    output_path,
    config_path,
    sample_rate
):
    with open(config_path) as r:
        config = json.load(r)

    include_sources = config["include_sources"]
    include_labels = config["include_labels"]
    max_length = config["max_length"]
    records = read_jsonl(input_path, sample_rate)

    texts, sources = list(), dict()
    for record in records:
        text = preprocess_text(record["text"])
        label = record["label"]
        source = record["source"]
        if "aug" in record:
            continue
        if label not in include_labels:
            continue
        if source not in include_sources:
            continue
        if len(text) > max_length:
            continue
        sources[text] = source
        texts.append(text)

    write_jsonl([{"text": t, "source": sources[t]} for t in texts], output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()
    main(**vars(args))
