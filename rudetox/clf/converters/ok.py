import argparse
import json

from rudetox.util.io import write_jsonl


def main(
    input_file,
    output_file
):
    records = []
    bad_labels = ("__label__INSULT", "__label__OBSCENITY", "__label__THREAT")
    with open(input_file) as r:
        for line in r:
            labels = line.split(" ")[0].split(",")
            text = " ".join(line.split(" ")[1:]).strip()
            label = 0
            for lbl in labels:
                if lbl in bad_labels:
                    label = 1
            records.append({
                "text": text,
                "label": label,
                "source": "ok"
            })
    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
