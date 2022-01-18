import argparse
import copy

import torch
from transformers import AutoModelForTokenClassification, AutoModelForMaskedLM
from transformers import AutoTokenizer, pipeline, BasicTokenizer
from tqdm import tqdm

from util.io import read_jsonl, write_jsonl
from util.dl import pipe_predict
from util.text import preprocess_text


class WordFiller:
    def __init__(self, model_name, device, top_k=6):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.word_tokenizer = BasicTokenizer(do_lower_case=False)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.top_k = top_k

    def replace_words(self, words, replace_indices):
        assert replace_indices
        replace_word_idx = replace_indices.pop()
        hyps = self._search(words, replace_word_idx, replace_indices)
        assert len(hyps) > 1
        return [self._words_to_sentence(hyp_words) for hyp_words in hyps]

    def _search(self, words, replace_word_idx, remaining_replace_indices):
        new_words = copy.deepcopy(words)
        new_words[replace_word_idx] = "[MASK]"

        tokens = self._words_to_tokens(new_words).to(self.model.device)
        mask_token_index = tokens.tolist().index(self.tokenizer.mask_token_id)

        logits = self.model(input_ids=tokens.unsqueeze(0)).logits.squeeze(0)
        mask_token_logits = logits[mask_token_index]
        replacing_tokens = torch.topk(mask_token_logits, self.top_k, dim=0).indices.tolist()

        top_hyps = [words]
        for token_id in replacing_tokens:
            if token_id in self.tokenizer.all_special_ids:
                continue
            tokens[mask_token_index] = token_id
            hyp = self.tokenizer.decode(tokens[1:-1], skip_special_tokens=False)
            hyp_words = self.word_tokenizer.tokenize(hyp)
            if len(hyp_words) != len(words):
                continue
            top_hyps.append(hyp_words)

        if not remaining_replace_indices:
            return top_hyps

        next_remaining_replace_indices = copy.deepcopy(remaining_replace_indices)
        next_replace_word_idx = next_remaining_replace_indices.pop()
        new_hyps = []
        for hyp_words in top_hyps:
            new_hyps.extend(self._search(hyp_words, next_replace_word_idx, next_remaining_replace_indices))
        return new_hyps

    def _words_to_tokens(self, words):
        tokens = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt"
        ).input_ids
        return tokens.squeeze(0)

    def _words_to_sentence(self, words):
        tokens = self._words_to_tokens(words)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def main(
    model_name,
    filler_model_name,
    input_path,
    output_path,
    text_field,
    sample_rate
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_num = 0 if device == "cuda" else -1

    records = read_jsonl(input_path, sample_rate)
    texts = [r[text_field][:500] for r in records]

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
        filler = WordFiller(filler_model_name, device)

    tokens_predictions, tokens_scores = pipe_predict(texts, pipe)
    word_tokenizer = BasicTokenizer(do_lower_case=False)
    records = []
    for text, predictions, scores in zip(texts, tokens_predictions, tokens_scores):
        text = preprocess_text(text)
        words = word_tokenizer.tokenize(text)
        tokens_encoded = tokenizer.encode_plus(text)
        tokens = tokens_encoded["input_ids"]

        scores = scores[:len(tokens)]
        predictions = predictions[:len(tokens)]
        for i, (label, score) in enumerate(zip(predictions, scores)):
            if label == 0:
                scores[i] = 1.0 - scores[i]

        rm_indices = torch.argsort(torch.tensor(scores), descending=True).tolist()
        for it in range(1, len(rm_indices)):
            target_tokens = [token for i, token in enumerate(tokens) if i not in rm_indices[:it]]
            target = tokenizer.decode(target_tokens, skip_special_tokens=True)
            records.append({"target": target, "source": text, "type": "marker_delete"})

        true_rm_indices = [idx for idx in rm_indices if scores[idx] >= 0.4]
        word_rm_indices = [tokens_encoded.token_to_word(token_index) for token_index in true_rm_indices]
        word_rm_indices = list(sorted({idx for idx in word_rm_indices if idx is not None}))
        print("Text:", text, "; bad words: ", word_rm_indices)

        if not filler or not (1 <= len(word_rm_indices) <= 5) or tokenizer.unk_token_id in tokens:
            continue

        hyps = filler.replace_words(words, word_rm_indices)
        print("Hyps:", len(hyps))

        records.extend([{"target": hyp, "source": text, "type": "condbert"} for hyp in hyps])

    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--filler-model-name", type=str, default=None)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
