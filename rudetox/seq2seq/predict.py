import argparse
import json

import torch
import razdel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util.io import read_jsonl, write_jsonl
from util.dl import gen_batch, set_random_seed
from ranker import Ranker


def predict(
    model_name,
    input_file,
    sample_rate,
    output_file,
    batch_size,
    max_source_tokens_count,
    seed,
    no_repeat_ngram_size,
    repetition_penalty,
    length_penalty,
    num_beams,
    num_return_sequences,
    early_stopping,
    source_field,
    ranker_config
):
    set_random_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    ranker = None
    if ranker_config:
        ranker_config = json.loads(ranker_config)
        ranker = Ranker(**ranker_config)

    output_texts, scores = [], []
    records = list(read_jsonl(input_file, sample_rate))
    for batch in tqdm(gen_batch(records, batch_size)):
        texts = [r[source_field] for r in batch]
        input_ids = tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        output_ids = model.generate(
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )
        output_ids = output_ids.reshape((len(batch), num_return_sequences, output_ids.size(1)))

        for text, sample_output_ids in zip(texts, output_ids):
            targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in sample_output_ids]
            best_target = targets[0]
            if ranker:
                best_target, best_target_scores = ranker(text, targets)
                scores.append(best_target_scores)
            output_texts.append(best_target)

    for target, s, r in zip(output_texts, scores, records):
        r["target"] = target
        r["scores"] = s

    if ranker:
        print("Style:", sum([s["style"] for s in scores]) / len(output_texts))
        print("Fluency:", sum([s["fluency"] for s in scores]) / len(output_texts))
        print("Sim:", sum([s["sim"] for s in scores]) / len(output_texts))

    write_jsonl(records, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens-count", type=int, default=600)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    parser.add_argument("--source-field", type=str, default="text")
    parser.add_argument("--ranker-config", type=str, default=None)
    args = parser.parse_args()
    predict(**vars(args))
