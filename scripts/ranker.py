import argparse
from collections import defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer
from tqdm import tqdm

from util.io import read_jsonl, write_jsonl
from util.dl import run_clf

STYLE_MODEL = "SkolkovoInstitute/russian_toxicity_classifier"
MEANING_MODEL = "cointegrated/LaBSE-en-ru"
FLUENCY_MODEL = "SkolkovoInstitute/rubert-base-corruption-detector"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Ranker:
    def __init__(
        self,
        style_model_name=STYLE_MODEL,
        meaning_model_name=MEANING_MODEL,
        fluency_model_name=FLUENCY_MODEL,
        device=DEVICE,
        good_style_label=0
    ):
        self.style_model = AutoModelForSequenceClassification.from_pretrained(style_model_name)
        self.style_model = self.style_model.to(device)
        self.style_tokenizer = AutoTokenizer.from_pretrained(style_model_name)
        self.good_style_label = good_style_label

        self.meaning_model = AutoModel.from_pretrained(meaning_model_name)
        self.meaning_model = self.meaning_model.to(device)
        self.meaning_tokenizer = AutoTokenizer.from_pretrained(meaning_model_name)

        self.fluency_model = AutoModelForSequenceClassification.from_pretrained(fluency_model_name)
        self.fluency_model = self.fluency_model.to(device)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained(fluency_model_name)

        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    @staticmethod
    def calc_embedding(texts, tokenizer, model):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(model.device)
        out = model(**inputs)
        embeddings = out.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings

    def eval_style(self, texts):
        return run_clf(texts, self.style_tokenizer, self.style_model)

    def eval_fluency(self, texts):
        return run_clf(texts, self.fluency_tokenizer, self.fluency_model)

    def __call__(self, source, targets):
        source_style_label = self.eval_style([source])[0]
        source_fluency_label = self.eval_fluency([source])[0]

        style_labels = self.eval_style(targets)
        good_style_targets = [t for l, t in zip(style_labels, targets) if l == self.good_style_label]
        has_good_style = len(good_style_targets) != 0
        if not has_good_style:
            good_style_targets = targets

        fluency_labels = self.eval_fluency(good_style_targets)
        fluent_targets = [t for l, t in zip(fluency_labels, good_style_targets) if l == 1]
        has_fluent = len(fluent_targets) != 0
        if not has_fluent:
            fluent_targets = good_style_targets

        targets = fluent_targets
        sources = [source for _ in range(len(targets))]
        source_embeddings = self.calc_embedding(sources, self.meaning_tokenizer, self.meaning_model)
        target_embeddings = self.calc_embedding(targets, self.meaning_tokenizer, self.meaning_model)
        scores = self.cos(source_embeddings, target_embeddings)
        max_score, best_index = torch.max(scores, 0)
        metrics = {
            "source_style": source_style_label,
            "source_fluency": source_fluency_label,
            "style": self.good_style_label if has_good_style else 1 - self.good_style_label,
            "fluency": int(has_fluent),
            "sim": max_score.item()
        }
        return targets[best_index], metrics


def main(
    source_field,
    target_field,
    input_path,
    output_path,
    sample_rate
):
    ranker = Ranker()
    records = read_jsonl(input_path, sample_rate)
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
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--source-field", type=str, default="source")
    parser.add_argument("--target-field", type=str, default="target")
    args = parser.parse_args()
    main(**vars(args))
