import sys
from collections import Counter
from util.io import read_jsonl, write_jsonl


def is_good_record(r, skip_reasons):
    scores = r["scores"]
    target = r["target"]
    if scores["style"] > 0.2:
        skip_reasons["style"] += 1
        return False
    if scores["source_style"] == 0:
        skip_reasons["source_style"] += 1
        return False
    if scores["fluency"] < 0.4:
        skip_reasons["fluency"] += 1
        return False
    if scores["sim"] < 0.8:
        skip_reasons["sim"] += 1
        return False
    if scores["chrf"] < 0.4:
        skip_reasons["chrf"] += 1
        return False
    bad_substrings = ("творен", "паден", "греба", "пипс")
    for ss in bad_substrings:
        if ss in target.lower():
            skip_reasons["bad_substrings"] += 1
            return False
    return True


input_path = sys.argv[1]
output_path = sys.argv[2]
records = read_jsonl(input_path)
skip_reasons = Counter()
records = [r for r in records if is_good_record(r, skip_reasons)]
print(skip_reasons.most_common())
write_jsonl(records, output_path)
