import argparse
import os

from util import read_jsonl, write_jsonl


def main(
    input_files,
    output_file
):
    records = []
    for f in input_files:
        assert os.path.exists(f)
        records += read_jsonl(f)
    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs='+')
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
