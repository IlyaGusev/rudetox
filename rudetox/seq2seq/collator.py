from typing import List, Dict

import numpy as np
import torch
from transformers import BatchEncoding


class T5MLMDataCollator:
    def __init__(
        self,
        tokenizer,
        model,
        pad_token_id,
        input_length,
        noise_density,
        mean_noise_span_length
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.input_length = input_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.pad_token_id = pad_token_id

    def collate_batch(self, examples):
        examples = [e for e in examples]
        result = examples[0].new_full((len(examples), self.input_length), self.tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            result[i, :example.size(0)] = example
        return result

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        keys = examples[0].keys()
        batch = BatchEncoding(
            {k: self.collate_batch([examples[i][k] for i in range(len(examples))]) for k in keys}
        )

        input_ids = batch["input_ids"]
        lengths = batch["attention_mask"].sum(axis=1)
        batch_size, max_length = input_ids.size(0), input_ids.size(1)

        mask_indices = [torch.tensor(self.random_spans_noise_mask(length-1)) for length in lengths]
        mask_indices = self.collate_batch(mask_indices)
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.numpy().astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.numpy().astype(np.int8))

        new_input_ids = torch.tensor(self.filter_input_ids(input_ids.numpy(), input_ids_sentinel))
        new_mask = (new_input_ids != self.tokenizer.pad_token_id).long()
        labels = torch.tensor(self.filter_input_ids(input_ids.numpy(), labels_sentinel))
        #print()
        #print(self.tokenizer.decode(new_input_ids[0]).replace("<pad>", ""))
        #print(self.tokenizer.decode(labels[1]).replace("<pad>", ""))
        labels[labels == self.tokenizer.pad_token_id] = -100
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
        # TODO: WTF masks

        #print(labels[1])
        return {
            "input_ids": new_input_ids,
            "labels": labels,
            "attention_mask": new_mask,
            "decoder_input_ids": decoder_input_ids
        }

    def create_sentinel_ids(self, mask_indices):
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        batch_size = input_ids.shape[0]
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        max_length = 0
        for i in range(input_ids_full.shape[0]):
            sample = input_ids_full[i]
            filtered_sample = sample[sample > 0]
            if filtered_sample[-1] != self.tokenizer.eos_token_id:
                filtered_sample = np.append(filtered_sample, self.tokenizer.eos_token_id)
            max_length = max(max_length, len(filtered_sample))
            input_ids_full[i, :] = np.zeros(input_ids_full.shape[1])
            input_ids_full[i, :len(filtered_sample)] = filtered_sample
        return input_ids_full

    def random_spans_noise_mask(self, length):
        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]
