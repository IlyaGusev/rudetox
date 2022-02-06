import sys
import random
import os

from rudetox.util.io import write_jsonl

input_path = sys.argv[1]
assert os.path.exists(input_path)

source_path = sys.argv[2]
assert os.path.exists(source_path)

output_path = sys.argv[3]

mode = sys.argv[4]
assert mode in ("sim", "style")

random.seed(43)
records = []
with open(input_path) as r, open(source_path) as s:
    next(s)
    for line, source_line in zip(r, s):
        line = line.strip()
        source_line = source_line.strip().split("\t")[0]
        if mode == "sim":
            records.append({
                "first_text": source_line,
                "second_text": line
            })
        else:
            records.append({
                "text": line
            })

random.shuffle(records)
write_jsonl(records[:200], output_path)
