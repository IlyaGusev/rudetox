import argparse
import csv
import json


def main(
    input_file,
    output_file,
    include_neutrals,
    source_only
):
    neutral_keys = ("neutral_comment1", "neutral_comment2", "neutral_comment3")
    records = []
    with open(input_file, "r") as r:
        header = next(r).strip().split("\t")
        reader = csv.reader(r, delimiter="\t", quotechar='"')
        for row in reader:
            record = dict(zip(header, row))
            toxic = record["toxic_comment"]
            neutrals = [record.get(key) for key in neutral_keys if record.get(key)]
            if neutrals:
                for neutral in neutrals:
                    records.append({
                        "source": toxic,
                        "target": neutral,
                        "is_toxic": True
                    })
            else:
                records.append({
                    "source": toxic,
                    "is_toxic": True
                })
            if include_neutrals:
                for n1 in neutrals:
                    for n2 in neutrals:
                        r = {
                            "source": n1,
                            "target": n2,
                            "is_toxic": False
                        }
                        records.append(r)
    if source_only:
        sources = [records[0]["source"]]
        for r in records[1:]:
            source = r["source"]
            if source == sources[-1]:
                continue
            sources.append(source)
        records = [{"source": s} for s in sources]

    with open(output_file, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--source-only", action="store_true", default=False)
    parser.add_argument("--include-neutrals", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
