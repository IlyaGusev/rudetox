import argparse
from collections import Counter

import razdel
import numpy as np
from faiss import GpuIndexFlatIP, StandardGpuResources
from tqdm import tqdm

from rudetox.util.io import read_jsonl, write_jsonl


def main(
    input_path,
    output_path,
    bad_vocab_path,
    sample_rate
):
    vocab = set()
    with open(bad_vocab_path) as r:
        for line in r:
            vocab.add(line.strip())

    records = read_jsonl(input_path, sample_rate)

    filtered_records = []
    non_toxic_bad_words_count = 0
    long_texts_count = 0
    short_texts_count = 0
    for record in tqdm(records):
        label = record["label"]
        text = record["text"]
        if len(text) > 500:
            long_texts_count += 1
            continue
        if len(text) < 7:
            short_texts_count += 1
            continue

        tokens = [t.text.lower() for t in razdel.tokenize(text)]
        has_bad_token = False
        for token in tokens:
            if token in vocab:
                has_bad_token = True
        if has_bad_token and label != 1:
            non_toxic_bad_words_count += 1
            continue

        filtered_records.append(record)

    print("Short texts:", short_texts_count)
    print("Long texts:", long_texts_count)
    print("Non toxic texts with bad words:", non_toxic_bad_words_count)
    print("Before undup:", len(filtered_records))
    filtered_records = list({r["text"]: r for r in filtered_records}.values())

    d = len(filtered_records[0]["embedding"])
    embeddings_matrix = np.zeros((len(filtered_records), d)).astype('float32')
    for i, r in enumerate(filtered_records):
        embeddings_matrix[i] = r["embedding"]

    gpu_res = StandardGpuResources()
    ann_index = GpuIndexFlatIP(gpu_res, d)
    ann_index.add(embeddings_matrix)
    remove_indices = set()
    for first_index, r in tqdm(enumerate(filtered_records)):
        if first_index in remove_indices:
            continue
        embedding = np.array(r.pop("embedding"), dtype="float32")
        embedding = np.expand_dims(embedding, axis=0)
        distances, indices = ann_index.search(embedding, 30)
        for distance, second_index in zip(distances[0][1:], indices[0][1:]):
            first_label = filtered_records[first_index]["label"]
            second_label = filtered_records[second_index]["label"]
            if distance >= 0.9 and first_label == second_label and second_index > first_index:
                remove_indices.add(second_index)

    final_records = [r for i, r in enumerate(filtered_records) if i not in remove_indices]
    print("After undup:", len(final_records))
    write_jsonl(final_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    parser.add_argument("--sample-rate", default=1.0, type=float)
    parser.add_argument("--bad-vocab-path", required=True, type=str)
    args = parser.parse_args()
    main(**vars(args))
