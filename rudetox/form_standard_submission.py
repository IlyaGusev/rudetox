import argparse
import csv


def main(input_file, output_file):
    neutral_keys = ("neutral_comment1", "neutral_comment2", "neutral_comment3")
    with open(input_file, "r") as r, open(output_file, "w") as w:
        header = next(r).split("\t")
        reader = csv.reader(r, delimiter="\t", quotechar='"')
        for row in reader:
            record = dict(zip(header, row))
            neutrals = [record.get(key) for key in neutral_keys if record.get(key)]
            neutral = neutrals[0]
            w.write(neutral.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
