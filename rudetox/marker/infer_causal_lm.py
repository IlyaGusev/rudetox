import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from rudetox.util.io import read_jsonl


def calc_tokens_loss(model, input_ids):
    logits = model(
        input_ids=input_ids,
        return_dict=True
    ).logits
    probs = torch.softmax(logits, dim=1)
    tokens_probs = [p[token_id].item() for p, token_id in zip(probs, input_ids[1:])]
    tokens_loss = torch.log(torch.tensor(tokens_probs))
    return tokens_loss


def main(
    toxic_model_name,
    non_toxic_model_name,
    input_path
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    toxic_model = AutoModelForCausalLM.from_pretrained(toxic_model_name).to(device)
    non_toxic_model = AutoModelForCausalLM.from_pretrained(non_toxic_model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(toxic_model_name)

    records = list(read_jsonl(input_path))
    for r in tqdm(records):
        if r["label"] == 0:
            continue
        input_ids = [tokenizer.bos_token_id]
        input_ids += tokenizer(
            r["text"],
            add_special_tokens=False,
            max_length=100,
            padding=False,
            truncation=True
        ).input_ids
        input_ids.append(tokenizer.eos_token_id)
        input_ids = torch.LongTensor(input_ids).to(device)

        toxic_tokens_loss = calc_tokens_loss(toxic_model, input_ids)[:-1]
        non_toxic_tokens_loss = calc_tokens_loss(non_toxic_model, input_ids)[:-1]

        print()
        print(r["text"])
        scores = non_toxic_tokens_loss / toxic_tokens_loss
        for score, token_id in zip(scores, input_ids[1:]):
            score = score.item()
            token = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            print(token, score, score > 3.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--toxic-model-name", type=str, required=True)
    parser.add_argument("--non-toxic-model-name", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
