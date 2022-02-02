import sys

from rudetox.util.io import read_jsonl

input_path = sys.argv[1]
output_path = sys.argv[2]

records = read_jsonl(input_path)
with open(output_path, "w") as w:
    for r in records:
        w.write(r["target"].strip() + "\n")
