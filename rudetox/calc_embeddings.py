import argparse
import json
import copy

from tqdm import tqdm

from rudetox.util.dl import Embedder, gen_batch
from rudetox.util.io import read_jsonl


def main(
    input_path,
    output_path,
    batch_size,
    sample_rate,
    text_field
):
    model = Embedder(batch_size=batch_size)
    docs = list(read_jsonl(input_path, sample_rate))
    with open(output_path, "w") as w:
        for batch in tqdm(gen_batch(docs, batch_size)):
            texts = [r[text_field] for r in batch]
            embeddings = model(texts)
            for r, embedding in zip(batch, embeddings):
                r["embedding"] = [float(v) for v in embedding]
                w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--sample-rate', type=float, default=1.0)
    parser.add_argument('--text-field', type=str, default="text")
    args = parser.parse_args()
    main(**vars(args))
