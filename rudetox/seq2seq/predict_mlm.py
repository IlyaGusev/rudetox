import argparse
import copy

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import gen_batch, set_random_seed


def predict(
    input_path,
    output_path,
    text_field,
    output_field,
    model_name,
    sample_rate,
    batch_size,
    max_source_tokens_count,
    max_target_tokens_count,
    seed,
    no_repeat_ngram_size,
    repetition_penalty,
    length_penalty,
    num_beams,
    num_return_sequences,
    early_stopping
):
    set_random_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)

    outputs = []
    records = list(read_jsonl(input_path, sample_rate))
    for batch in tqdm(gen_batch(records, batch_size)):
        texts = [r[text_field] for r in batch]
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
            num_return_sequences=num_return_sequences,
            max_length=max_target_tokens_count,
            eos_token_id=model.config.eos_token_id
        )
        output_ids = output_ids.reshape((len(batch), num_return_sequences, output_ids.size(1)))

        for sample_input_ids, sample_output_ids in zip(input_ids, output_ids):
            sample_input_ids = sample_input_ids.tolist()
            if model.config.eos_token_id not in sample_input_ids:
                continue
            sample_input_ids = sample_input_ids[:sample_input_ids.index(model.config.eos_token_id)]
            for ids in sample_output_ids:
                ids = ids.tolist()
                if model.config.eos_token_id not in ids:
                    continue
                ids = ids[:ids.index(model.config.eos_token_id)]
                fixed_input_ids = copy.copy(sample_input_ids)

                print()
                old_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
                print(old_text)
                filler = tokenizer.decode(ids, skip_special_tokens=False)
                print(filler)

                current_extra_id = tokenizer.encode("<extra_id_0>")[0]
                next_extra_id = current_extra_id - 1
                while next_extra_id in ids and current_extra_id in fixed_input_ids:
                    current_pos = ids.index(current_extra_id)
                    next_pos = ids.index(next_extra_id)
                    filler = ids[current_pos + 1: next_pos]
                    while current_extra_id in filler:
                        filler = filler[filler.index(current_extra_id) + 1:]
                    template_pos = fixed_input_ids.index(current_extra_id)
                    fixed_input_ids = fixed_input_ids[:template_pos] + filler + fixed_input_ids[template_pos+1:]
                    current_extra_id = next_extra_id
                    next_extra_id -= 1

                new_text = tokenizer.decode(fixed_input_ids, skip_special_tokens=False).replace("<pad>", "")
                print(new_text)
                if "extra_id" in new_text:
                    continue
                assert "extra_id" not in new_text
                outputs.append(new_text)
    output_records = []
    for i, output in enumerate(outputs):
        record_index = i // num_return_sequences
        new_record = copy.copy(records[record_index])
        new_record[output_field] = output
        output_records.append(new_record)
    write_jsonl(output_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--text-field", type=str, required=True)
    parser.add_argument("--output-field", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-source-tokens-count", type=int, default=100)
    parser.add_argument("--max-target-tokens-count", type=int, default=100)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--early-stopping", action="store_true", default=False)
    args = parser.parse_args()
    predict(**vars(args))
