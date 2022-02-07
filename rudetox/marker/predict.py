import argparse
import copy
import random

import torch
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline, BasicTokenizer
from tqdm import tqdm

from rudetox.marker.compute_tags import tags_to_template, MASK_TEMPLATE
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
    seed
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    random.seed(seed)
    records = list(read_jsonl(input_path, sample_rate))
    texts = [preprocess_text(r[text_field]) for r in records]

    model = AutoModelForTokenClassification.from_pretrained(tagger_model_name)
    tokenizer = AutoTokenizer.from_pretrained(tagger_model_name)
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device_num
    )
    marker_tags, marker_scores = pipe_predict(texts, pipe)

    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
    for text_num, (record, predictions, scores) in tqdm(enumerate(zip(records, marker_tags, marker_scores))):
        text = preprocess_text(record[text_field])
        tokens = tokenizer(text).input_ids
        predictions = predictions[:len(tokens)]
        tokens = tokens[:len(predictions)]
        template = tags_to_template(tokens, predictions, tokenizer)
        if "extra_id" in template:
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
                num_beams=5,
                max_length=300
            )
            fillers = gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            mask_count = template.count("extra_id")
            for mask_num in range(mask_count):
                current_mask = MASK_TEMPLATE.format(mask_num)
                next_mask = MASK_TEMPLATE.format(mask_num + 1)
                start_index = fillers.find(current_mask) + len(current_mask)
                end_index = fillers.find(next_mask)
                template = template.replace(current_mask, fillers[start_index:end_index])
        record["target"] = template
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tagger-model-name", type=str, required=True)
    parser.add_argument("--gen-model-name", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
