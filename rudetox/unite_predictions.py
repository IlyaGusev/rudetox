import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from rudetox.ranker import Ranker


def main(
    predictions,
    source_path,
    output_path,
    ranker_config_path
):
    all_targets = []
    for path in predictions:
        texts = []
        with open(path) as r:
            for line in r:
                line = line.strip()
                texts.append(line)
        all_targets.append(texts)
    n = len(all_targets[0])
    ttargets = []
    for i in range(n):
        ttargets.append([all_targets[j][i] for j in range(len(all_targets))])

    sources = []
    with open(source_path) as r:
        next(r)
        for line in r:
            sources.append(line.strip())

    assert os.path.exists(ranker_config_path)
    with open(ranker_config_path) as r:
        config = json.load(r)
    ranker = Ranker(**config)


    cnt = Counter()
    scores = []
    with open(output_path, "w") as w:
        for source, targets in zip(sources, ttargets):
            best_target, s = ranker(source, targets)
            scores.append(s)
            print(best_target)
            cnt[targets.index(best_target)] += 1
            w.write(best_target.strip() + "\n")

    print(cnt.most_common())
    print("Style:", sum([s["style"] for s in scores]) / n)
    print("Fluency:", sum([s["fluency"] for s in scores]) / n)
    print("Sim:", sum([s["sim"] for s in scores]) / n)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs='+')
    parser.add_argument("--source-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--ranker-config-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
