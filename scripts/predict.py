import argparse

import torch
import razdel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForSequenceClassification

from util import read_jsonl, gen_batch, set_random_seed


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
    source_field,
    clf_name
):
    set_random_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    records = list(read_jsonl(input_file))
    if clf_name:
        clf_tokenizer = AutoTokenizer.from_pretrained(clf_name)
        clf_model = AutoModelForSequenceClassification.from_pretrained(clf_name).to(device)
        def infer_clf(text):
            input_ids = clf_tokenizer.encode(text, return_tensors="pt").to(device)
            clf_prob = clf_model(input_ids).logits[0][0].item()
            return clf_prob

    summaries = []
    count_clf_ok = 0
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
            if clf_name:
                best_summary = None
                max_clf_prob = -100.0
                for ids in sample_output_ids:
                    summary = tokenizer.decode(ids, skip_special_tokens=True)
                    clf_prob = infer_clf(summary)
                    if clf_prob > max_clf_prob:
                        max_clf_prob = clf_prob
                        best_summary = summary
                if max_clf_prob > 0.0:
                    count_clf_ok += 1
                text_clf_prob = infer_clf(text)
                summaries.append(best_summary if text_clf_prob < 0.0 else text)
                continue
            summary = tokenizer.decode(sample_output_ids[0], skip_special_tokens=True)
            summaries.append(summary)

    if clf_name:
        print(count_clf_ok / len(summaries))

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
    parser.add_argument("--clf-name", type=str, default=None)
    args = parser.parse_args()
    predict(**vars(args))
