import argparse
from collections import defaultdict
import os
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from tqdm import tqdm
from nltk.translate.chrf_score import sentence_chrf

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import Classifier, Embedder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Ranker:
    def __init__(
        self,
        style_model_name,
        meaning_model_name,
        fluency_model_name,
        use_clf_filter,
        weights,
        device=DEVICE,
        invert_style=False
    ):
        self.style_model = Classifier(style_model_name, device=device)
        self.fluency_model = Classifier(fluency_model_name, device=device)
        self.meaning_model = Embedder(meaning_model_name, device=device)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        self.invert_style = invert_style
        self.use_clf_filter = use_clf_filter
        self.weights = weights

    def eval_style(self, texts):
        return self.style_model(texts)

    def eval_fluency(self, texts):
        return self.fluency_model(texts)

    def eval_sim(self, source, targets):
        sources = [source for _ in range(len(targets))]
        source_embeddings = self.meaning_model(sources)
        target_embeddings = self.meaning_model(targets)
        sim_scores = self.cos(source_embeddings, target_embeddings).tolist()
        return sim_scores

    def eval_chrf(self, source, targets):
        sources = [source for _ in range(len(targets))]
        chrf_scores = [sentence_chrf(source, target, beta=1.0) for source, target in zip(sources, targets)]
        return chrf_scores

    @staticmethod
    def scores_to_ranks(scores, descending=False):
        indices = torch.argsort(torch.tensor(scores), descending=descending).tolist()
        ranks = {int(index): float(rank) / max(len(indices) - 1, 1) for rank, index in enumerate(indices)}
        return ranks

    def __call__(self, source, targets):
        source_style_label = int(self.eval_style([source])[0][0])
        source_fluency_label = int(self.eval_fluency([source])[0][0])

        if self.use_clf_filter:
            good_style_label = 1 if self.invert_style else 0
            style_labels, _ = self.eval_style(targets)
            good_style_targets = [tgt for lbl, tgt in zip(style_labels, targets) if lbl == good_style_label]
            has_good_style = len(good_style_targets) != 0
            if not has_good_style:
                good_style_targets = targets

            fluency_labels, _ = self.eval_fluency(good_style_targets)
            fluent_targets = [tgt for lbl, tgt in zip(fluency_labels, good_style_targets) if lbl == 1]
            has_fluent = len(fluent_targets) != 0
            if not has_fluent:
                fluent_targets = good_style_targets

            targets = fluent_targets

        _, style_scores = self.eval_style(targets)
        _, fluent_scores = self.eval_fluency(targets)
        sim_scores = self.eval_sim(source, targets)
        chrf_scores = self.eval_chrf(source, targets)

        style_ranks = self.scores_to_ranks(style_scores, descending=not self.invert_style)
        fluent_ranks = self.scores_to_ranks(fluent_scores)
        sim_ranks = self.scores_to_ranks(sim_scores)
        chrf_ranks = self.scores_to_ranks(chrf_scores)
        all_ranks = [style_ranks, fluent_ranks, sim_ranks, chrf_ranks]

        final_ranks = []
        for index in range(len(targets)):
            final_rank = sum(ranks[index] * self.weights[fidx] for fidx, ranks in enumerate(all_ranks))
            final_ranks.append(final_rank)

        best_index = torch.max(torch.tensor(final_ranks), 0)[1].item()

        metrics = {
            "source_style": source_style_label,
            "source_fluency": source_fluency_label,
            "style": style_scores[best_index],
            "style_rank": style_ranks[best_index],
            "fluency": fluent_scores[best_index],
            "fluency_rank": fluent_ranks[best_index],
            "sim": sim_scores[best_index],
            "sim_rank": sim_ranks[best_index],
            "chrf": chrf_scores[best_index],
            "chrf_rank": chrf_ranks[best_index],
            "weights": list(self.weights),
            "final_rank": final_ranks[best_index]
        }
        return targets[best_index], metrics


def main(
    source_field,
    target_field,
    input_path,
    output_path,
    sample_rate,
    invert_style,
    config_path
):
    assert os.path.exists(config_path)
    with open(config_path) as r:
        config = json.load(r)

    ranker = Ranker(**config)
    records = list(read_jsonl(input_path, sample_rate))
    mapping = defaultdict(set)
    for r in records:
        mapping[r[source_field]].add(r[target_field])
    output_records = []
    for source, targets in tqdm(mapping.items()):
        targets = list(targets)
        best_target, scores = ranker(source, targets)
        output_records.append({
            "source": source,
            "target": best_target,
            "scores": scores
        })
    write_jsonl(output_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--config-path", type=str, default="configs/ranker.json")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--source-field", type=str, default="source")
    parser.add_argument("--target-field", type=str, default="target")
    args = parser.parse_args()
    main(**vars(args))
