import json
import random
import os


def read_jsonl(file_path, sample_rate=1.0):
    assert os.path.exists(file_path)
    with open(file_path) as r:
        for line in r:
            if random.random() <= sample_rate:
                yield json.loads(line)


def write_jsonl(records, file_path):
    with open(file_path, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")
