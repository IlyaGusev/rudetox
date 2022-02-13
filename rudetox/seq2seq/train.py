import argparse
import random
import json

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments, logging
from transformers import EncoderDecoderModel, AutoModelForSeq2SeqLM

from rudetox.seq2seq.dataset import Seq2seqDataset
from rudetox.util.io import read_jsonl
from rudetox.util.dl import set_random_seed, fix_tokenizer


def train(
    config_path,
    checkpoint,
    train_path,
    val_path,
    train_sample_rate,
    val_sample_rate,
    out_dir,
    report_to,
    seed,
    source_field,
    target_field,
    style_field,
    override_base_model
):
    set_random_seed(seed)
    logging.set_verbosity_info()
    with open(config_path, "r") as r:
        config = json.load(r)

    model_type = config["model_type"]
    assert model_type in ("encoder_decoder", "seq2seq_lm")
    model_name = config["model_name"]
    if override_base_model:
        model_name = override_base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False, strip_accents=False)
    tokenizer = fix_tokenizer(tokenizer)

    # Data preparation
    train_records = list(read_jsonl(train_path))
    val_records = list(read_jsonl(val_path))
    random.shuffle(train_records)

    dataset_class = Seq2seqDataset
    max_source_tokens_count = config["max_source_tokens_count"]
    max_target_tokens_count = config["max_target_tokens_count"]
    train_dataset_args = {
        "original_records": train_records,
        "sample_rate": train_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_target_tokens_count": max_target_tokens_count,
        "source_field": source_field,
        "target_field": target_field,
        "style_field": style_field
    }
    val_dataset_args = {
        "original_records": val_records,
        "sample_rate": val_sample_rate,
        "tokenizer": tokenizer,
        "max_source_tokens_count": max_source_tokens_count,
        "max_target_tokens_count": max_target_tokens_count,
        "source_field": source_field,
        "target_field": target_field,
        "style_field": style_field
    }
    train_dataset = dataset_class(**train_dataset_args)
    val_dataset = dataset_class(**val_dataset_args)

    # Model loading
    if model_type == "encoder_decoder":
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name, model_name)
    elif model_type == "seq2seq_lm":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        assert False

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
    model.config.max_length = max_target_tokens_count
    if model_type == "causal_lm":
        model.config.max_length = max_target_tokens_count + max_source_tokens_count

    # Training
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
        eval_dataset=val_dataset
    )
    trainer.train(checkpoint)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--train-path", type=str, required=True)
    parser.add_argument("--val-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--train-sample-rate", type=float, default=1.0)
    parser.add_argument("--val-sample-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", type=str, default="none")
    parser.add_argument("--source-field", type=str, default="source")
    parser.add_argument("--target-field", type=str, default="target")
    parser.add_argument("--style-field", type=str, default=None)
    parser.add_argument("--override-base-model", type=str, default=None)
    args = parser.parse_args()
    train(**vars(args))
