import argparse
import copy
import random

import torch
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline, BasicTokenizer
from tqdm import tqdm

from rudetox.marker.util import MASK_TEMPLATE, token_labels_to_template, convert_template_to_t5
from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import pipe_predict
from rudetox.util.text import preprocess_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    tagger_model_name,
    gen_model_name,
    input_path,
    output_path,
    text_field,
    sample_rate,
    seed,
    num_beams,
    num_return_sequences
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    random.seed(seed)
    records = list(read_jsonl(input_path, sample_rate))
    texts = [preprocess_text(r[text_field]) for r in records]

    tagger_model = AutoModelForTokenClassification.from_pretrained(tagger_model_name)
    tagger_tokenizer = AutoTokenizer.from_pretrained(tagger_model_name)
    tagger_pipe = pipeline(
        "token-classification",
        model=tagger_model,
        tokenizer=tagger_tokenizer,
        framework="pt",
        device=device_num,
        aggregation_strategy="max"
    )
    predictions = tagger_pipe(texts, batch_size=1)
    #predictions = pipe_predict(texts, pipe)[0]

    tagger_only_count = 0
    new_records = list()
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    for text_num, (record, sample_predictions) in enumerate(zip(records, predictions)):
        print()
        text = preprocess_text(record[text_field])
        print(text)

        template = []
        for group in sample_predictions:
            tag = group["entity_group"]
            phrase = group["word"]
            pad_index = phrase.find(tagger_tokenizer.pad_token)
            if pad_index != -1:
                phrase = phrase[:pad_index]
            if tag == "delete":
                continue
            if tag == "replace":
                phrase = tagger_tokenizer.mask_token
            template.append(phrase.strip())
        template = " ".join(template)
        template = convert_template_to_t5(template, tagger_tokenizer.mask_token)
        #tokens = tokenizer(text).input_ids
        #predictions = predictions[:len(tokens)]
        #tokens = tokens[:len(predictions)]
        #template = token_labels_to_template(tokens, predictions, tokenizer)
        record["template"] = template
        print(template)

        if "extra_id" not in template:
            tagger_only_count += 1
            record["target"] = template
            new_records.append(record)
            continue

        input_ids = gen_tokenizer(
            text,
            text_pair=template,
            add_special_tokens=True,
            max_length=200,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        output_ids = gen_model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=300,
            repetition_penalty=2.5
        )
        for sample_output_ids in output_ids:
            new_record = copy.copy(record)
            fillers = gen_tokenizer.decode(sample_output_ids, skip_special_tokens=False)
            new_record["fillers"] = fillers
            print(fillers)

            mask_count = template.count("extra_id")
            target = template
            for mask_num in range(mask_count):
                current_mask = MASK_TEMPLATE.format(mask_num).strip()
                next_mask = MASK_TEMPLATE.format(mask_num + 1).strip()
                start_index = fillers.find(current_mask) + len(current_mask)
                end_index = fillers.find(next_mask)
                filler = fillers[start_index:end_index]
                target = target.replace(current_mask, filler)
            target = " ".join(target.split())
            target = target.replace(" ,", ",")
            print(target)

            new_record["target"] = target
            new_records.append(new_record)

    print("Tagger only: {}".format(tagger_only_count))
    write_jsonl(new_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagger-model-name", type=str, required=True)
    parser.add_argument("--gen-model-name", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--num-return-sequences", type=int, default=10)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
