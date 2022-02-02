import argparse
import os
import json
import random

import numpy as np
from faiss import Kmeans, IndexFlatL2
from tqdm import tqdm
import torch
from scipy.stats import entropy

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import gen_batch
from rudetox.clf.train import train as train_clf


def initial_clustering(
    records,
    embedding_field,
    n_clusters,
    nredo=2,
    niter=30
):
    d = len(records[0][embedding_field])
    features_matrix = np.zeros((len(records), d), dtype=np.float32)
    for i, record in enumerate(records):
        features_matrix[i] = record[embedding_field]
    print("Matrix {}x{}".format(*features_matrix.shape))

    clustering = Kmeans(d=d, k=n_clusters, verbose=True, nredo=nredo, niter=niter)
    clustering.train(features_matrix)

    index = IndexFlatL2(d)
    index.add(features_matrix)
    center_points = index.search(clustering.centroids, 1)[1]
    center_points = np.squeeze(center_points)

    best_records = [records[index] for index in center_points]
    return best_records


def infer_clf(
    records,
    model,
    tokenizer,
    config,
    models_count,
    batch_size,
    text_field
):
    model.train() # Monte-Carlo Dropout

    output_records = []
    for batch in tqdm(gen_batch(records, batch_size)):
        texts = [r[text_field] for r in batch]
        inputs = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=config["max_tokens"],
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        num_labels = model.num_labels
        all_scores = torch.zeros((len(batch), models_count, num_labels))
        with torch.no_grad():
            for model_num in range(models_count):
                output = model(**inputs)
                logits = output.logits
                scores = torch.softmax(logits, dim=1).cpu()
                all_scores[:, model_num, :] = scores

        for sample_num in range(len(batch)):
            sample = batch[sample_num]
            sample_scores = all_scores[sample_num]

            avg_scores = torch.mean(sample_scores, dim=0).tolist()
            entropy_over_avg = float(entropy(avg_scores))

            entropies = [float(entropy(scores)) for scores in sample_scores]
            avg_entropy = float(np.mean(entropies))

            bald_score = entropy_over_avg - avg_entropy

            sample["entropy"] = entropy_over_avg
            sample["avg_entropy"] = avg_entropy
            sample["bald"] = bald_score
            sample["scores"] = avg_scores
            output_records.append(sample)
    return output_records


def main(
    input_path,
    output_path,
    embedding_field,
    initial_records_count,
    initial_records_path,
    clf_config_path,
    seed,
    infer_models_count,
    infer_batch_size,
    infer_samples_count,
    save_top_samples_count,
    text_field,
    res_field,
    final_size
):
    records = list(read_jsonl(input_path))
    assert records
    print("{} records read".format(len(records)))

    if os.path.exists(initial_records_path):
        initial_records = list(read_jsonl(initial_records_path))
    else:
        initial_records = initial_clustering(records, embedding_field, initial_records_count)
        write_jsonl(initial_records, initial_records_path)
    print("{} initial records".format(len(initial_records)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(clf_config_path, "r") as r:
        clf_config = json.load(r)

    labeled_records = initial_records
    epochs_count = (final_size - initial_records_count) // save_top_samples_count
    for epoch_num in range(epochs_count):
        print()
        print("Epoch {}".format(epoch_num))
        random.shuffle(labeled_records)
        labeled_keys = {r[text_field] for r in labeled_records}
        unlabeled_records = [r for r in records if r[text_field] not in labeled_keys]
        random.shuffle(unlabeled_records)
        unlabeled_records = unlabeled_records[:infer_samples_count]

        n = len(labeled_records)
        border = n * 9 // 10
        train_records = labeled_records[:border]
        val_records = labeled_records[border:]
        model, tokenizer = train_clf(
            train_records,
            val_records,
            config=clf_config,
            seed=seed,
            text_field=text_field,
            res_field=res_field,
            device=device
        )
        scored_records = infer_clf(
            unlabeled_records,
            model,
            tokenizer,
            clf_config,
            infer_models_count,
            infer_batch_size,
            text_field
        )
        scored_records.sort(key=lambda x: x["entropy"], reverse=True)
        for r in scored_records:
            r["epoch"] = epoch_num
        labeled_records.extend(scored_records[:save_top_samples_count])
        del model

    write_jsonl(labeled_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--clf-config-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-records-count", type=int, default=10000)
    parser.add_argument("--initial-records-path", type=str, default="initial.jsonl")
    parser.add_argument("--infer-models-count", type=int, default=5)
    parser.add_argument("--infer-samples-count", type=int, default=30000)
    parser.add_argument("--save-top-samples-count", type=int, default=3000)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--res-field", type=str, default="label")
    parser.add_argument("--embedding-field", type=str, default="embedding")
    parser.add_argument("--final-size", type=int, default=40000)
    parser.add_argument("--infer-batch-size", type=int, default=32)
    args = parser.parse_args()
    main(**vars(args))
