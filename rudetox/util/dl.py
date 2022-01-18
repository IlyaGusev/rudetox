import random
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "bos_token" in special_tokens:
        special_tokens["sep_token"] = special_tokens["bos_token"]

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def pipe_predict(data, pipe):
    raw_preds = pipe(data, batch_size=64)
    label2id = pipe.model.config.label2id
    y_pred, scores = [], []
    if isinstance(raw_preds[0], list):
        for sample in raw_preds:
            length = max([s["index"] for s in sample]) + 1
            sample_y_pred = [-1 for _ in range(length)]
            sample_y_scores = [-100.0 for _ in range(length)]
            for s in sample:
                sample_y_pred[s["index"]] = label2id[s["entity"]]
                sample_y_scores[s["index"]] = s["score"]
            y_pred.append(sample_y_pred)
            scores.append(sample_y_scores)
    else:
        y_pred = np.array([label2id[sample["label"]] for sample in raw_preds])
        scores = np.array([sample["score"] for sample in raw_preds])
    return y_pred, scores


def run_clf(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
        labels = torch.argmax(logits, dim=1)
    return labels


class Classifier:
    def __init__(
        self,
        model_name,
        batch_size=64,
        device=DEVICE
    ):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device
        self.batch_size = batch_size

    def __call__(self, texts):
        labels = []
        for batch in gen_batch(texts, batch_size=self.batch_size):
            labels.extend(run_clf(batch, self.tokenizer, self.model))
        return labels
