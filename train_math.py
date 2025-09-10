import os
import argparse
import json
from pathlib import Path
import torch
import transformers
from datasets import load_dataset, load_dataset_builder, concatenate_datasets
import numpy as np
import random
from transformers import EarlyStoppingCallback

# ---------------------------
# Repro
# ---------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(1234)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Llama2-ish chat wrappers
# ---------------------------
SYSTEM_TOKEN = "<<SYS>>"
END_SYS_TOKEN = "<</SYS>>"
INST_TOKEN = "[INST]"
END_INST = "[/INST]"

def format_chat_prompt(system: str, user: str, assistant: str = None) -> str:
    """
    Wraps messages in Llama2-style [INST] chat format.
    """
    prefix = f"{INST_TOKEN} {SYSTEM_TOKEN}\n{system}\n{END_SYS_TOKEN}\n{user} {END_INST}"
    if assistant is not None:
        return f"{prefix}\n{assistant}"
    return prefix

def tokenize_chat(tokenizer, text: str, cutoff_len: int, add_eos: bool):
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    # Ensure python lists (some tokenizers return lists already)
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    if add_eos and (len(input_ids) < cutoff_len):
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (len(input_ids) == 0 or input_ids[-1] != eos_id):
            input_ids = input_ids + [eos_id]
            attention_mask = attention_mask + [1]

    return {"input_ids": input_ids, "attention_mask": attention_mask}

def build_tokens(tokenizer, system, user, assistant, args):
    chat = format_chat_prompt(system, user, assistant)
    tokens = tokenize_chat(tokenizer, chat, args.cutoff_len, args.add_eos_token)
    input_ids = tokens["input_ids"].copy()
    attention_mask = tokens["attention_mask"]
    if assistant is not None and not args.train_on_inputs:
        # mask the user/system part with -100 so loss only on assistant
        prompt = format_chat_prompt(system, user)
        pr_tokens = tokenize_chat(tokenizer, prompt, args.cutoff_len, args.add_eos_token)
        prefix_len = len(pr_tokens["input_ids"])
        labels = [-100] * prefix_len + input_ids[prefix_len:]
    else:
        labels = input_ids.copy()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ---------------------------
# Dataset processor (MATH)
# ---------------------------
def process_math(dp, tokenizer, args):
    """
    Each dp typically has fields: 'problem', 'solution', 'level', 'type', ...
    We'll train supervised to reproduce the *worked solution*.
    """
    system = args.system_prompt
    problem = dp.get("problem", "")
    # Some variants store solution as string; ensure it's str
    solution = dp.get("solution", "")
    if isinstance(solution, (list, tuple)):
        solution = "\n".join(map(str, solution))
    solution = str(solution)

    user = (
        "Solve the following math problem. Show clear, step-by-step reasoning and end with the final answer.\n\n"
        f"Problem:\n{problem}\n\nAnswer:"
    )
    assistant = " " + solution.strip()
    return build_tokens(tokenizer, system, user, assistant, args)

# ---------------------------
# Main
# ---------------------------
def main(args):
    os.environ["WANDB_PROJECT"] = args.wandb_project
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Load model & tokenizer
    model_id = args.model_path if args.model_path else args.base_model
    if model_id is None:
        raise ValueError("You must provide either --base_model or --model_path")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True,cache_dir='/data')
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id,cache_sir='/data')

    model.to(device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    # ---------------------------
    # Load ALL subjects from EleutherAI/hendrycks_math
	# ---------------------------

    from datasets import get_dataset_config_names

    ds_name = "EleutherAI/hendrycks_math"
    all_subjects = get_dataset_config_names(ds_name)  # e.g. ['algebra', 'geometry', ...]
    if args.include_subjects:
        requested = [s.strip() for s in args.include_subjects.split(",") if s.strip()]
        missing = [s for s in requested if s not in all_subjects]
        if missing:
            raise ValueError(f"Subjects not found: {missing}\nAvailable: {sorted(all_subjects)}")
        subjects = requested
    else:
        subjects = all_subjects

    print(f"Loading {len(subjects)} MATH subjects:", subjects)

    train_shards = []
    val_shards = []

    for name in subjects:
        # Many configs have train/test. If no validation, we'll split off a small part of train for val.
        raw = load_dataset(ds_name, name,cache_dir='/data')
        if "validation" in raw:
            tr = raw["train"]
            val = raw["validation"]
        else:
            # reserve args.val_size_per_subject examples for validation from train (or from test if no train)
            if "train" in raw:
                base = raw["train"]
            elif "test" in raw:
                base = raw["test"]
            else:
                raise ValueError(f"Unexpected splits for subject {name}: {raw}")

            # If the subject is too small, safe-guard the split size
            n_val = min(args.val_size_per_subject, len(base))
            split = base.train_test_split(test_size=n_val, seed=42)
            tr, val = split["train"], split["test"]

        # map -> tokenized features
        tr_proc = tr.map(lambda x: process_math(x, tokenizer, args), remove_columns=tr.column_names)
        val_proc = val.map(lambda x: process_math(x, tokenizer, args), remove_columns=val.column_names)

        train_shards.append(tr_proc)
        val_shards.append(val_proc.select(range(min(len(val_proc), args.max_val_per_subject))))

    # Concatenate across subjects
    train_dataset = concatenate_datasets(train_shards).shuffle(seed=42)
    val_dataset = concatenate_datasets(val_shards).shuffle(seed=42)

    # ---------------------------
    # HF Trainer
    # ---------------------------
    # Choose precision based on hardware; keep flag to override
    fp16 = args.fp16 and (device == "cuda")
    bf16 = (not fp16) and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

    grad_accum = max(1, args.batch_size // max(1, args.micro_batch_size))

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,  # usually "eval_loss"
        greater_is_better=False,
        report_to="wandb" if args.report_to_wandb else "none",
        run_name=args.run_name if args.report_to_wandb else None,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, padding=True, return_tensors="pt"
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SFT Chat-style Finetuning on EleutherAI/hendrycks_math (all subjects)")
    # model
    p.add_argument("--base_model", type=str, default=None, help="HF base model (e.g., meta-llama/Llama-2-7b-hf)")
    p.add_argument("--model_path", type=str, default=None, help="Existing checkpoint to continue finetuning")
    p.add_argument("--output_dir", type=str, default="./sft-math", help="Save directory")

    # batching / training
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--cutoff_len", type=int, default=2048)

    # precision & misc
    p.add_argument("--fp16", action="store_true", help="Use fp16 if available (bf16 auto if Ampere+ and fp16 not set)")
    p.add_argument("--train_on_inputs", action="store_true", help="Include prompt tokens in loss")
    p.add_argument("--add_eos_token", action="store_true")

    # logging / wandb
    p.add_argument("--run_name", type=str, default="math-sft-all-subjects")
    p.add_argument("--wandb_project", type=str, default="finetuning")
    p.add_argument("--report_to_wandb", action="store_true")

    # checkpointing & eval
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    p.add_argument("--early_stopping_patience", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=1000)

    # prompts
    p.add_argument("--system_prompt", type=str, default="You are a brilliant math tutor. Be rigorous yet clear.")

    # dataset options
    p.add_argument("--include_subjects", type=str, default=None,
                   help="Comma-separated subset of subjects to train on. Default: use all.")
    p.add_argument("--val_size_per_subject", type=int, default=200,
                   help="How many examples to hold out per subject when no val split exists.")
    p.add_argument("--max_val_per_subject", type=int, default=400,
                   help="Upper cap of validation examples per subject to keep eval small.")

    args = p.parse_args()
    main(args)
