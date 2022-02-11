import argparse
import random
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, logging
from transformers import AutoModelForSeq2SeqLM

from rudetox.seq2seq.dataset import TextDataset
from rudetox.seq2seq.collator import T5MLMDataCollator
from rudetox.util.io import read_jsonl
from rudetox.util.dl import set_random_seed, fix_tokenizer


def train(
    config_path,
    checkpoint,
    input_path,
    sample_rate,
    out_dir,
    report_to,
    seed,
    text_field,
    override_base_model,
    val_part
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_path, "r") as r:
        config = json.load(r)

    model_name = config["model_name"]
    if override_base_model:
        model_name = override_base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    tokenizer = fix_tokenizer(tokenizer)

    # Data loading
    records = list(read_jsonl(input_path, sample_rate))
    random.shuffle(records)

    border = int(len(records) * (1.0 - val_part))
    train_records = records[:border]
    val_records = records[border:]

    # Data preparation
    max_source_tokens_count = config["max_source_tokens_count"]
    train_dataset = TextDataset(
        train_records,
        tokenizer=tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        text_field=text_field
    )
    val_dataset = TextDataset(
        val_records,
        tokenizer=tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        text_field=text_field
    )

    # Model loading
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Special tokens
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        if bos_candidate is not None:
            model.config.bos_token_id = bos_candidate
            model.config.decoder_start_token_id = bos_candidate
            break
    assert model.config.bos_token_id is not None

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    # Default model generation params
    model.config.num_beams = 5
    model.config.max_length = max_source_tokens_count

    # Training
    noise_density = config["noise_density"]
    mean_noise_span_length = config["mean_noise_span_length"]
    data_collator = T5MLMDataCollator(
        tokenizer=tokenizer,
        model=model,
        input_length=max_source_tokens_count,
        noise_density=noise_density,
        mean_noise_span_length=mean_noise_span_length,
        pad_token_id=model.config.pad_token_id
    )

    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    num_train_epochs = config["num_train_epochs"]

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=report_to
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    trainer.train(checkpoint)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--override-base-model", type=str, default=None)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-part", type=float, default=0.1)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--text-field", type=str, required=True)
    args = parser.parse_args()
    train(**vars(args))
