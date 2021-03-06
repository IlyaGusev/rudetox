import argparse
import json

import torch
from transformers import FSMTForConditionalGeneration, AutoTokenizer
from tqdm import tqdm

from rudetox.util.io import read_jsonl, write_jsonl
from rudetox.util.dl import gen_batch
from rudetox.util.text import preprocess_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BacktransParaphraser:
    def __init__(
        self,
        forward_model_name,
        backward_model_name,
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
                "target": target,
                "type": "backtrans"
            })
        return outputs

    def is_good_hyp(self, text):
        has_latin = any("a" <= ch.lower() <= "z" for ch in text)
        if has_latin:
            return False
        if len(text) > 200:
            return False
        return True


def main(
    input_path,
    output_path,
    sample_rate,
    num_beams,
    batch_size,
    text_field,
    forward_model_name,
    backward_model_name
):
    paraphraser = BacktransParaphraser(
        num_beams=num_beams,
        batch_size=batch_size,
        forward_model_name=forward_model_name,
        backward_model_name=backward_model_name
    )
    records = read_jsonl(input_path, sample_rate)
    texts = [preprocess_text(record[text_field]) for record in records]
    output_records = paraphraser(texts)
    write_jsonl(output_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--forward-model-name", type=str, required=True)
    parser.add_argument("--backward-model-name", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
