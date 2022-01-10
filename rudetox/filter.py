import sys
from util.io import read_jsonl, write_jsonl


def is_good_record(r):
    scores = r["scores"]
    target = r["target"]
    if scores["style"] == 1:
        return False
    if scores["source_style"] == 0:
        return False
    if scores["fluency"] == 0:
        return False
    if scores["sim"] < 0.85:
        return False
    if scores["chrf"] < 0.4:
        return False
    if "творен" in target.lower():
        return False
    if "паден" in target.lower():
        return False
    if "греба" in target.lower():
        return False
    if "пипс" in target.lower():
        return False
    return True


input_path = sys.argv[1]
output_path = sys.argv[2]
records = read_jsonl(input_path)

records = [r for r in records if is_good_record(r)]
write_jsonl(records, output_path)
