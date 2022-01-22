import argparse
import os
from collections import defaultdict, Counter

import toloka.client as toloka
import pandas as pd
from nltk.metrics.agreement import AnnotationTask
from crowdkit.aggregation import DawidSkene

from rudetox.util.io import write_jsonl
from rudetox.crowd.util import get_pool, read_token, read_pools_ids, get_key


def aggregate(
    records,
    res_field,
    key_fields,
    min_agreement=0.7
):
    records.sort(key=lambda x: x["assignment_id"])

    results = defaultdict(list)
    for r in records:
        results[get_key(r, key_fields)].append(r[res_field])

    data = {get_key(r, key_fields): r for r in records}

    votes, votes_distribution, res_distribution = dict(), Counter(), Counter()
    for key, res in results.items():
        res_count = Counter(res)
        overlap = len(res)
        res_win, votes_win = res_count.most_common(1)[0]
        res_distribution[res_win] += 1
        votes_part = float(votes_win) / overlap
        votes_distribution[votes_part] += 1
        votes[key] = votes_part
        data[key].update({
            res_field: res_win,
            "agreement": votes_part
        })

    answers = [(str(hash(get_key(r, key_fields))), r[res_field], r["worker_id"]) for r in records]
    answers_df = pd.DataFrame(answers, columns=["task", "label", "performer"])
    proba = DawidSkene(n_iter=20).fit_predict_proba(answers_df)
    labels = proba.idxmax(axis=1)
    for key in data:
        ds_key = str(hash(key))
        label = labels[ds_key]
        confidence = proba.loc[ds_key, label]
        data[key].update({
            "ds_{}".format(res_field): label,
            "ds_confidence": confidence
        })

    total_samples = len(data)
    print("Total: ", total_samples)
    print("Aggreements:")
    sum_agreement = 0
    for v, sample_count in sorted(votes_distribution.items(), reverse=True):
        print("{}: {}".format(v, sample_count))
        sum_agreement += sample_count * v
    print("Average agreement:", sum_agreement / total_samples)
    print()

    print("Results:")
    for res, cnt in res_distribution.items():
        print("{}: {}".format(res, cnt))
    print()

    answers = [(r["worker_id"], get_key(r, key_fields), r[res_field]) for r in records]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha: {}".format(t.alpha()))

    answers = [
        (r["worker_id"], get_key(r, key_fields), r[res_field])
        for r in records if votes[get_key(r, key_fields)] >= min_agreement
    ]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha, border {}: {}".format(min_agreement, t.alpha()))
    print()

    data = {key: r for key, r in data.items()}
    return data


def main(
    token_path,
    agg_output,
    raw_output,
    pools_path,
    input_fields,
    res_field,
    key_fields
):
    key_fields = key_fields.split(",")
    input_fields = input_fields.split(",")

    toloka_client = toloka.TolokaClient(read_token(token_path), 'PRODUCTION')
    pool_ids = read_pools_ids(pools_path)

    records = []
    for pool_id in pool_ids:
        records.extend(get_pool(pool_id, toloka_client))

    agg_records = aggregate(records, res_field, key_fields)
    agg_records = list(agg_records.values())
    agg_records.sort(key=lambda x: x["agreement"], reverse=True)
    agg_header = [res_field, "agreement", "ds_result", "ds_confidence"] + input_fields
    agg_records = [{key: r[key] for key in agg_header} for r in agg_records]
    write_jsonl(agg_records, agg_output)

    raw_records = records
    raw_header = [res_field, "worker_id", "assignment_id"] + input_fields
    raw_records = [{key: r[key] for key in raw_header} for r in raw_records]
    write_jsonl(raw_records, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fields", type=str, default="text")
    parser.add_argument("--token-path", type=str, default="~/.toloka/personal_token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-path", type=str, required=True)
    parser.add_argument("--res-field", type=str, default="result")
    parser.add_argument("--key-fields", type=str, default="text")
    args = parser.parse_args()
    main(**vars(args))
