import argparse
import random

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from rudetox.marker.util import MASK_TEMPLATE
from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import pipe_predict, gen_batch


def main(
    model_name,
    input_path,
    output_path,
    max_source_tokens_count,
    sample_rate,
    batch_size,
    layer_number
):
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, output_attentions=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    records = list(read_jsonl(input_path, sample_rate))
    output_texts = []
    for batch in gen_batch(records, batch_size):
        batch = [r["text"] for r in batch]
        batch_size = len(batch)
        inputs = tokenizer(
            batch,
            add_special_tokens=True,
            max_length=max_source_tokens_count,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        lengths = [torch.sum(i).item() for i in inputs["attention_mask"]]

        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions
            attentions = attentions[layer_number]
            attentions = attentions.max(1)[0] # max pool by heads
            cls_attentions = attentions[:, 0] # cls token
            for i, (attn, length, input_ids) in enumerate(zip(cls_attentions, lengths, inputs["input_ids"])):
                current_attn = attn[:length][1:-1].softmax(-1)
                avg_value = current_attn.mean().item()
                top_masks = (current_attn > avg_value).nonzero() + 1
                top_masks = top_masks.view(-1).cpu().tolist()
                current_sent = input_ids[:length]

                for index in top_masks:
                    current_token = tokenizer.decode(current_sent[index], skip_special_tokens=False).strip()
                    good_tokens = (",", ".", "!", "?", "не", "так", "это", "а", "у", "тебе", "[MASK]")
                    if current_token in good_tokens:
                        continue

                    current_sent[index] = tokenizer.mask_token_id
                    for next_index in range(index + 1, len(current_sent)):
                        next_token = tokenizer.decode(current_sent[next_index], skip_special_tokens=False).strip()
                        if not next_token.startswith("#"):
                            break
                        current_sent[next_index] = tokenizer.mask_token_id

                    if not current_token.startswith("#"):
                        continue
                    for prev_index in range(index - 1, 0, -1):
                        prev_token = tokenizer.decode(current_sent[prev_index], skip_special_tokens=False).strip()
                        current_sent[prev_index] = tokenizer.mask_token_id
                        if not prev_token.startswith("#"):
                            break

                squeezed_sent, prev_index = [], None
                for index in current_sent:
                    if index == tokenizer.mask_token_id and prev_index == tokenizer.mask_token_id:
                        continue
                    prev_index = index
                    squeezed_sent.append(index)
                template = tokenizer.decode(squeezed_sent[1:-1], skip_special_tokens=False).strip()

                current_pos, mask_num = 0, 0
                mask_pos = template.find(tokenizer.mask_token, current_pos)
                while mask_pos != -1:
                    end_mask_pos = mask_pos + len(tokenizer.mask_token)
                    template = template[:mask_pos] + MASK_TEMPLATE.format(mask_num) + template[end_mask_pos:]
                    template = " ".join(template.split())
                    current_pos = end_mask_pos
                    mask_pos = template.find(tokenizer.mask_token, current_pos)
                    mask_num += 1
                output_texts.append(template)

    for r, output in zip(records, output_texts):
        r["masked"] = output
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--max-source-tokens-count", type=int, default=100)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    # still the magical layer number of the Russian grand elves
    parser.add_argument("--layer-number", type=int, default=7)
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
