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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEQ2SEQ_REPLACER = "cointegrated/rut5-base"
DEFAULT_MLM_REPLACER = "DeepPavlov/rubert-base-cased-conversational"


class OneWordReplacer:
    def __init__(self, model_name, device, top_k):
        self.model_name = model_name
        self.device = device
        self.top_k = top_k

    def __call__(self, words, replace_word_idx):
        raise NotImplementedError


class Seq2seqOneWordReplacer(OneWordReplacer):
    '''
    Based on David Dale example: https://colab.research.google.com/drive/174nvR0LkggK7SN5E9j9Huiu_kLRoPGQm
    '''
    def __init__(
        self,
        top_k,
        model_name=DEFAULT_SEQ2SEQ_REPLACER,
        device=DEVICE,
        mask_token_id=29999,
        eos_token_id=29998,
        max_length=10,
        encoder_no_repeat_ngram_size=3,
        repetition_penalty=10.0
    ):
        super().__init__(model_name, device, top_k)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.mask_token_id = mask_token_id
        self.mask_token = self.tokenizer.decode([self.mask_token_id])

        self.eos_token_id = eos_token_id
        self.eos_token = self.tokenizer.decode([self.eos_token_id])
        self.max_length = max_length
        self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty

    def __call__(self, words, replace_word_idx):
        new_words = copy.deepcopy(words)
        new_words[replace_word_idx] = self.mask_token
        tokens = words_to_tokens(self.tokenizer, new_words).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=tokens.unsqueeze(0),
                repetition_penalty=self.repetition_penalty,
                encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
                max_length=self.max_length,
                eos_token_id=self.eos_token_id,
                num_beams=self.top_k,
                num_return_sequences=self.top_k
            )

        replacements = []
        for ids in output_ids:
            replacement = self.tokenizer.decode(ids, skip_special_tokens=True)
            replacement = replacement.replace(self.mask_token, "").replace(self.eos_token, "")
            for ch in (".", ",", "(", ")", "-", "?", "!"):
                replacement = replacement.replace(ch, " ")
            replacement = replacement.strip()
            if "<" in replacement or len(replacement) <= 1:
                continue
            replacements.append(replacement)
        replacements = list(set(replacements))
        return replacements


class MLMOneWordReplacer(OneWordReplacer):
    def __init__(
        self,
        top_k,
        model_name=DEFAULT_MLM_REPLACER,
        device=DEVICE,
        mask_token_id=103
    ):
        super().__init__(model_name, device, top_k)

        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.mask_token_id = mask_token_id
        self.mask_token = self.tokenizer.decode([self.mask_token_id])

    def __call__(self, words, replace_word_idx):
        new_words = copy.deepcopy(words)
        new_words[replace_word_idx] = self.mask_token
        tokens = words_to_tokens(self.tokenizer, new_words).to(self.model.device)

        mask_token_index = (tokens == self.tokenizer.mask_token_id).nonzero().item()
        logits = self.model(input_ids=tokens.unsqueeze(0)).logits.squeeze(0)
        mask_token_logits = logits[mask_token_index]
        mask_token_logits[self.tokenizer.all_special_ids] = -100.0
        replacing_tokens = torch.topk(mask_token_logits, self.top_k, dim=0).indices
        replacing_tokens = [self.tokenizer.decode(token, skip_special_tokens=True) for token in replacing_tokens]
        return replacing_tokens


class WordFiller:
    def __init__(self, model_type, **kwargs):
        self.word_tokenizer = BasicTokenizer(do_lower_case=False)
        self.model_type = model_type
        if model_type == "mlm":
            self.replacer = MLMOneWordReplacer(**kwargs)
        elif model_type == "seq2seq":
            self.replacer = Seq2seqOneWordReplacer(**kwargs)

    def replace_words(self, words, replace_indices):
        assert replace_indices

        print("INPUT:", " ".join(words), replace_indices)
        # Start recursive search
        hyps = self._search(words, replace_indices)

        print("OUTPUT:")
        for hyp_words, _ in hyps:
            print(" ".join(hyp_words))

        # Words to sentences
        hyps = list({words_to_sentence(self.replacer.tokenizer, hyp_words) for hyp_words, _ in hyps})
        return hyps

    def _search(self, words, replace_indices):
        if not replace_indices:
            return [(words, [])]

        # Infer filler model, get candidates for mask
        replace_word_idx = replace_indices.pop()
        replacements = self.replacer(words, replace_word_idx)
        new_hyps = [(words, replace_indices)]

        for replacement in replacements:
            new_words = self.word_tokenizer.tokenize(replacement)
            words_diff = len(new_words) - 1
            hyp_words = words[:replace_word_idx] + new_words + words[replace_word_idx + 1:]
            hyp_indices = [idx if idx < replace_word_idx else idx + words_diff for idx in replace_indices]
            #if hyp_indices:
            #    print("STEP:", " ".join(hyp_words), hyp_indices)
            new_hyps.extend(self._search(hyp_words, hyp_indices))
        return new_hyps


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
    filler_model_type,
    top_k
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
    if filler_model_type:
        filler = WordFiller(
            model_type=filler_model_type,
            device=device,
            top_k=top_k
        )

    records = []
    word_tokenizer = BasicTokenizer(do_lower_case=False)
    marker_tags, marker_scores = pipe_predict(texts, pipe)
    for text_num, (text, predictions, scores) in tqdm(enumerate(zip(texts, marker_tags, marker_scores))):
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
        word_replace_indices = [idx for idx in word_replace_indices if idx is not None and len(words[idx]) > 1]
        word_replace_indices = get_first_elements(word_replace_indices)
        bad_words = [words[idx] for idx in word_replace_indices]
        #print()
        #print("Num: {}, text: {}, bad_words: {}".format(text_num, text, bad_words))

        can_replace = (1 <= len(word_replace_indices) <= max_replace_words)
        if not filler or not can_replace or tokenizer.unk_token_id in tokens:
            #print("Skip replace step")
            continue

        hyps = filler.replace_words(words, word_replace_indices[::-1])
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
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filler-model-name", type=str, default=None)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--filler-mask-token", type=str, default="[MASK]")
    parser.add_argument("--filler-model-type", type=str, default="seq2seq", choices=("mlm", "seq2seq"))
    args = parser.parse_args()
    main(**vars(args))
