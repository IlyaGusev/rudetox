import argparse
import copy
import random

import torch
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, pipeline, BasicTokenizer
from tqdm import tqdm

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import pipe_predict, words_to_tokens, words_to_sentence
from rudetox.util.helpers import get_first_elements
from rudetox.util.text import preprocess_text


class WordFiller:
    def __init__(self, model_name, device, mask_token, model_type, top_k=10, eos_token_id=250098):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word_tokenizer = BasicTokenizer(do_lower_case=False)
        self.model_type = model_type
        if model_type == "mlm":
            self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        elif model_type == "seq2seq":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.top_k = top_k
        self.mask_token = mask_token
        self.eos_token_id = eos_token_id

    def replace_words(self, words, replace_indices):
        assert replace_indices

        # Start recursive search
        replace_word_idx = replace_indices.pop()
        hyps = self._search(words, replace_word_idx, replace_indices)

        # Words to sentences
        hyps = list({words_to_sentence(self.tokenizer, hyp_words) for hyp_words in hyps})
        return hyps

    def _search(self, words, replace_word_idx, remaining_replace_indices):
        # Infer filler model, get candidates for mask
        if self.model_type == "mlm":
            top_hyps = self.fill_mlm(words, replace_word_idx)
        elif self.model_type == "seq2seq":
            top_hyps = self.fill_seq2seq(words, replace_word_idx)

        if not remaining_replace_indices:
            return top_hyps

        # Search steps
        new_hyps = []
        next_replace_indices = copy.deepcopy(remaining_replace_indices)
        next_word_idx = next_replace_indices.pop()
        for hyp_words in top_hyps:
            new_hyps.extend(self._search(hyp_words, next_word_idx, next_replace_indices))
        return new_hyps

    def fill_mlm(self, words, replace_word_idx):
        new_words = copy.deepcopy(words)
        new_words[replace_word_idx] = self.mask_token
        tokens = words_to_tokens(self.tokenizer, new_words).to(self.model.device)

        mask_token_index = (tokens == self.tokenizer.mask_token_id).nonzero().item()
        logits = self.model(input_ids=tokens.unsqueeze(0)).logits.squeeze(0)
        mask_token_logits = logits[mask_token_index]
        mask_token_logits[self.tokenizer.all_special_ids] = -100.0
        replacing_tokens = torch.topk(mask_token_logits, self.top_k, dim=0).indices

        top_hyps = [words]
        for token_id in replacing_tokens:
            tokens[mask_token_index] = token_id
            hyp = self.tokenizer.decode(tokens, skip_special_tokens=True)
            hyp_words = self.word_tokenizer.tokenize(hyp)
            if len(hyp_words) != len(words):
                continue
            top_hyps.append(hyp_words)
        return top_hyps

    def fill_seq2seq(self, words, replace_word_idx):
        new_words = copy.deepcopy(words)
        new_words[replace_word_idx] = self.mask_token
        tokens = words_to_tokens(self.tokenizer, new_words).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=tokens.unsqueeze(0),
                repetition_penalty=10.0,
                max_length=12,
                eos_token_id=self.eos_token_id,
                num_beams=self.top_k,
                num_return_sequences=self.top_k
            )

        print("Orig:", " ".join(new_words))
        replacing_words = []
        for ids in output_ids:
            replacing_word = self.tokenizer.decode(ids, skip_special_tokens=True)
            replacing_word = replacing_word.replace("<extra_id_0>", "").replace("<extra_id_1>", "")
            for ch in (".", ",", "(", ")"):
                replacing_word = replacing_word.replace(ch, " ")
            replacing_word = replacing_word.strip()
            if "<" in replacing_word or len(replacing_word) <= 1 or " " in replacing_word:
                continue
            replacing_words.append(replacing_word)
        replacing_words = list(set(replacing_words))
        print(replacing_words)

        top_hyps = [words]
        for replacing_word in replacing_words:
            hyp_words = copy.deepcopy(words)
            hyp_words[replace_word_idx] = replacing_word
            hyp_words = " ".join(hyp_words).split()
            if len(hyp_words) != len(words):
                continue
            top_hyps.append(hyp_words)
        return top_hyps


def main(
    model_name,
    filler_model_name,
    input_path,
    output_path,
    text_field,
    sample_rate,
    replace_min_prob,
    max_replace_words,
    seed,
    filler_mask_token,
    filler_model_type
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    random.seed(seed)
    records = read_jsonl(input_path, sample_rate)
    texts = [r[text_field] for r in records]

    model = AutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pipe = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=device_num
    )

    filler = None
    if filler_model_name:
        filler = WordFiller(
            filler_model_name,
            device,
            mask_token=filler_mask_token,
            model_type=filler_model_type
        )

    records = []
    word_tokenizer = BasicTokenizer(do_lower_case=False)
    tokens_predictions, tokens_scores = pipe_predict(texts, pipe)
    for text_num, (text, predictions, scores) in enumerate(zip(texts, tokens_predictions, tokens_scores)):
        text = preprocess_text(text)
        words = word_tokenizer.tokenize(text)
        encoded = tokenizer.encode_plus(text, max_length=128)
        tokens = encoded["input_ids"]

        scores = scores[:len(tokens)]
        predictions = predictions[:len(tokens)]
        for i, (label, score) in enumerate(zip(predictions, scores)):
            if label == 0:
                scores[i] = 1.0 - scores[i]

        rm_indices = torch.argsort(torch.tensor(scores), descending=True).tolist()
        word_rm_indices = [encoded.token_to_word(token_index) for token_index in rm_indices]
        word_rm_indices = [word_index for word_index in word_rm_indices if word_index is not None]
        word_rm_indices = get_first_elements(word_rm_indices)

        for it in range(len(word_rm_indices)):
            target_words = [word for i, word in enumerate(words) if i not in word_rm_indices[:it]]
            target = words_to_sentence(tokenizer, target_words)
            records.append({"target": target, "source": text, "type": "marker_delete"})

        replace_indices = [idx for idx in rm_indices if scores[idx] >= replace_min_prob]
        word_replace_indices = [encoded.token_to_word(token_index) for token_index in replace_indices]
        word_replace_indices = list(sorted({idx for idx in word_replace_indices if idx is not None}))
        word_replace_indices = [idx for idx in word_replace_indices if len(words[idx]) > 1]
        bad_words = [words[idx] for idx in word_replace_indices]
        print()
        print("Num: {}, text: {}, bad_words: {}".format(text_num, text, bad_words))

        can_replace = (1 <= len(word_replace_indices) <= max_replace_words)
        if not filler or not can_replace or tokenizer.unk_token_id in tokens:
            print("Skip replace step")
            continue

        hyps = filler.replace_words(words, word_replace_indices)
        print("Hyps count: {}, random hyp: {}".format(len(hyps), random.choice(hyps)))
        records.extend([{"target": hyp, "source": text, "type": "condbert"} for hyp in hyps])

    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--replace-min-prob", type=float, default=0.4)
    parser.add_argument("--max-replace-words", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filler-model-name", type=str, default=None)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--filler-mask-token", type=str, default="[MASK]")
    parser.add_argument("--filler-model-type", type=str, default="mlm", choices=("mlm", "seq2seq"))
    args = parser.parse_args()
    main(**vars(args))
