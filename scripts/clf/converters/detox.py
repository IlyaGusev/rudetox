import argparse
import csv
import json

from util import write_jsonl


def main(
    input_file,
    output_file
):
    neutral_keys = ("neutral_comment1", "neutral_comment2", "neutral_comment3")
    records = []
    with open(input_file, "r") as r:
        header = next(r).split("\t")
        reader = csv.reader(r, delimiter="\t", quotechar='"')
        for row in reader:
            record = dict(zip(header, row))
            toxic = record["toxic_comment"]
            neutrals = [record.get(key) for key in neutral_keys if record.get(key)]
            records.append({
                "text": toxic,
                "label": 1,
                "source": "detox"
            })
            for neutral in neutrals:
                records.append({
                    "text": neutral,
                    "label": 0,
                    "source": "detox"
                })
    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
