import argparse
import csv
import json

from rudetox.util.io import write_jsonl


def main(
    input_file,
    output_file
):
    records = []
    with open(input_file) as r:
        reader = csv.reader(r)
        header = next(reader)
        for row in reader:
            r = dict(zip(header, row))
            records.append({
                "text": r["comment"].replace("\n", " ").replace("\t", " ").strip(),
                "label": int(float(r["toxic"])),
                "source": "2ch"
            })
    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
