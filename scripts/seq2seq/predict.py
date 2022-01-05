import argparse

import torch
import razdel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from util import read_jsonl, gen_batch, set_random_seed
from ranker import Ranker


def predict(
    model_name,
    input_file,
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
    source_field
):
    set_random_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    records = list(read_jsonl(input_file))
    ranker = Ranker()

    summaries = []
    count_style_ok = 0
    count_fluency_ok = 0
    sum_sim = 0.0
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
            best_target, scores = ranker(text, targets)
            count_style_ok += scores["style"]
            count_fluency_ok += scores["fluency"]
            sum_sim += scores["sim"]
            summaries.append(best_target)

    print("Style:", count_style_ok / len(summaries))
    print("Fluency:", count_fluency_ok / len(summaries))
    print("Sim:", sum_sim / len(summaries))

    with open(output_file, "w") as w:
        for s in summaries:
            w.write(s.strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
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
    args = parser.parse_args()
    predict(**vars(args))
