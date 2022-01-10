import argparse

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from util.io import read_jsonl, write_jsonl
from util.dl import gen_batch

PARAPHRASER_NAME = 'cointegrated/rut5-base-paraphraser'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Paraphraser:
    def __init__(
        self,
        model_name=PARAPHRASER_NAME,
        num_beams=10,
        encoder_no_repeat_ngram_size=4,
        batch_size=8,
        device=DEVICE
    ):
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size

    def __call__(self, texts):
        outputs = []
        for batch in tqdm(gen_batch(texts, batch_size=self.batch_size)):
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                padding=True
            ).to(self.model.device)

            output_ids = self.model.generate(
                **inputs,
                encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams
            )

            output_ids = output_ids.reshape((len(batch), self.num_beams, output_ids.size(1)))
            for text, sample_output_ids in zip(batch, output_ids):
                targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in sample_output_ids]
                for target in targets:
                    outputs.append({
                        "source": text,
                        "target": target
                    })
        return outputs


def main(
    input_path,
    output_path,
    num_beams,
    batch_size,
    sample_rate
):
    paraphraser = Paraphraser(
        num_beams=num_beams,
        batch_size=batch_size
    )

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
