import argparse
import csv
import json

TOXIC_KEY = "toxic_comment"
NEUTRAL_KEYS = ("neutral_comment1", "neutral_comment2", "neutral_comment3")

def main(
    input_file,
    output_file,
    include_auto_neutrals,
    include_auto_toxic,
    include_reverse
):
    records = []
    with open(input_file, "r") as r:
        header = next(r).strip().split("\t")
        reader = csv.reader(r, delimiter="\t", quotechar='"')
        for row in reader:
            record = dict(zip(header, row))
            toxic = record[TOXIC_KEY]
            neutrals = [record.get(key) for key in NEUTRAL_KEYS if record.get(key)]
            if neutrals:
                for neutral in neutrals:
                    records.append({
                        "source": toxic,
                        "target": neutral,
                        "style": 0
                    })
            else:
                records.append({
                    "source": toxic
                })
            if include_reverse:
                for neutral in neutrals:
                    records.append({
                        "source": neutral,
                        "target": toxic,
                        "style": 1
                    })
            if include_auto_neutrals:
                for n1 in neutrals:
                    for n2 in neutrals:
                        r = {
                            "source": n1,
                            "target": n2,
                            "style": 0
                        }
                        records.append(r)
            if include_auto_toxic:
                r = {
                    "source": toxic,
                    "target":toxic,
                    "style": 1
                }
                records.append(r)

    with open(output_file, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--include-auto-neutrals", action="store_true", default=False)
    parser.add_argument("--include-auto-toxic", action="store_true", default=False)
    parser.add_argument("--include-reverse", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
