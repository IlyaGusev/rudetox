import argparse
import json

import torch
from transformers import FSMTForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

from util.io import read_jsonl, write_jsonl
from util.dl import gen_batch

BACKWARD_MODEL = "facebook/wmt19-en-ru"
FORWARD_MODEL = "facebook/wmt19-ru-en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BacktransParaphraser:
    def __init__(
        self,
        forward_model_name=FORWARD_MODEL,
        backward_model_name=BACKWARD_MODEL,
        device=DEVICE,
        num_beams=5,
        block_ngram_size=4,
        batch_size=8
    ):
        self.forward_tokenizer = AutoTokenizer.from_pretrained(forward_model_name)
        self.forward_model = FSMTForConditionalGeneration.from_pretrained(forward_model_name)
        self.forward_model = self.forward_model.to(device)

        self.backward_tokenizer = AutoTokenizer.from_pretrained(backward_model_name)
        self.backward_model = FSMTForConditionalGeneration.from_pretrained(backward_model_name)
        self.backward_model = self.backward_model.to(device)

        self.device = device
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.block_ngram_size = block_ngram_size

    def translate(self, texts, model, tokenizer):
        outputs = []
        for batch in tqdm(gen_batch(texts, batch_size=self.batch_size)):
            inputs = tokenizer(
                batch,
                return_tensors='pt',
                padding=True
            ).to(model.device)
            output_ids = model.generate(
                **inputs,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams
            )
            output_ids = output_ids.reshape((len(batch), self.num_beams, output_ids.size(1)))
            for text, sample_output_ids in zip(batch, output_ids):
                targets = [tokenizer.decode(ids, skip_special_tokens=True) for ids in sample_output_ids]
                for target in targets:
                    outputs.append((text, target))
        return outputs

    def __call__(self, sources):
        translated_records = self.translate(sources, self.forward_model, self.forward_tokenizer)
        translated_records = list(set(translated_records))
        trans2source = {t: s for s, t in translated_records}

        translations = list({target for _, target in translated_records})
        backtranslated_records = self.translate(translations, self.backward_model, self.backward_tokenizer)
        backtrans2trans = {t: s for s, t in backtranslated_records}

        outputs = []
        for target, translation in backtrans2trans.items():
            source = trans2source[translation]
            if not self.is_good_hyp(target):
                continue
            if source == target:
                continue
            outputs.append({
                "source": source,
                "target": target
            })
        return outputs

    def is_good_hyp(self, text):
        has_latin = any("a" <= ch <= "z" for ch in text)
        if has_latin:
            return False
        if len(text) > 100:
            return False
        return True


def main(
    input_path,
    output_path,
    sample_rate,
    num_beams,
    batch_size
):
    paraphraser = BacktransParaphraser(num_beams=num_beams, batch_size=batch_size)
    records = read_jsonl(input_path, sample_rate)
    texts = [record["text"] for record in records]
    output_records = paraphraser(texts)
    write_jsonl(output_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    main(**vars(args))
