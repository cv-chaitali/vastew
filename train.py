


import os
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
import transformers
from datasets import load_dataset, concatenate_datasets
from transformers import EarlyStoppingCallback, TrainerCallback, AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login
login('')
# -----------------------
# Repro & CUDA setup
# -----------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seeds(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_TOKEN = "<<SYS>>"
END_SYS_TOKEN = "<</SYS>>"
INST_TOKEN = "[INST]"
END_INST = "[/INST]"

def format_chat_prompt(system: str, user: str, assistant: str = None) -> str:
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
    if add_eos and tokens["input_ids"] and tokens["input_ids"][-1] != tokenizer.eos_token_id and len(tokens["input_ids"]) < cutoff_len:
        tokens["input_ids"].append(tokenizer.eos_token_id)
        tokens["attention_mask"].append(1)
    return tokens

def build_tokens(tokenizer, system, user, assistant, args):
    chat = format_chat_prompt(system, user, assistant)
    tokens = tokenize_chat(tokenizer, chat, args.cutoff_len, args.add_eos_token)
    input_ids = tokens["input_ids"].copy()
    attention_mask = tokens["attention_mask"]
    if assistant is not None and not args.train_on_inputs:
        prompt = format_chat_prompt(system, user)
        pr_tokens = tokenize_chat(tokenizer, prompt, args.cutoff_len, args.add_eos_token)
        prefix_len = len(pr_tokens["input_ids"])
        labels = [-100] * prefix_len + input_ids[prefix_len:]
    else:
        labels = input_ids.copy()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def process_opencoder_educational(dp, tokenizer, args):
    system = args.system_prompt
    instruction = dp.get("instruction", "")
    user = f"Instruction: {instruction}"
    assistant = f"``````"  # original behavior preserved
    return build_tokens(tokenizer, system, user, assistant, args)

def process_opencoder_evol(dp, tokenizer, args):
    system = args.system_prompt
    instruction = dp.get("instruction", "")
    output = dp.get("output", "")
    user = f"Instruction: {instruction}"
    assistant = output
    return build_tokens(tokenizer, system, user, assistant, args)

def process_opencoder_mceval(dp, tokenizer, args):
    system = args.system_prompt
    instruction = dp.get("instruction", "")
    output = dp.get("output", "")
    user = f"Instruction: {instruction}"
    assistant = output
    return build_tokens(tokenizer, system, user, assistant, args)

def process_opencoder_package(dp, tokenizer, args):
    system = args.system_prompt
    instruction = dp.get("instruction", "")
    output = dp.get("output", "")
    user = f"Instruction: {instruction}"
    assistant = output
    return build_tokens(tokenizer, system, user, assistant, args)

# ---- Average of per-split losses ----
class AverageLossCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics", None)
        if not isinstance(metrics, dict):
            return
        split_losses = [
            v for k, v in metrics.items()
            if k.startswith("eval_") and k.endswith("_loss") and k != "eval_loss"
        ]
        if split_losses:
            metrics["eval_avg_loss"] = float(np.mean(split_losses))

# ---- Model loader: simple LoRA (no bitsandbytes, no FlashAttention2) ----
def load_tokenizer_and_model(args):
    model_id = args.model_path or args.base_model
    if model_id is None:
        raise ValueError("You must specify --base_model or --model_path")

    bf16_ok = torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if bf16_ok else torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,  # load to CPU, we .to(device) later
    )

    # Explicitly avoid FlashAttention2; use SDPA or default
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="sdpa",
            **model_kwargs, cache_dir="/root/.cache/huggingface/hub"
        )
    except TypeError:
        # Older transformers without attn_implementation arg
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs, cache_dir="/root/.cache/huggingface/hub")

    # LoRA wrap
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    return tokenizer, model

# ---- Merge helpers ----
def _dtype_from_str(s: str):
    s = s.lower()
    if s in {"fp16", "float16", "half"}: return torch.float16
    if s in {"bf16", "bfloat16"}: return torch.bfloat16
    if s in {"fp32", "float32", "full"}: return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def merge_and_save_adapters(adapters_dir: str, base_model_id: str, tokenizer, merged_output_dir: str, merge_dtype: str = "float16", merge_on_gpu: bool = False):
    torch_dtype = _dtype_from_str(merge_dtype)
    print(f"[MERGE] Loading base '{base_model_id}' in {merge_dtype} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if merge_on_gpu else None,
    )

    print(f"[MERGE] Loading LoRA adapters from '{adapters_dir}' ...")
    merged = PeftModel.from_pretrained(base, adapters_dir)
    print("[MERGE] Merging LoRA weights into base ...")
    merged = merged.merge_and_unload()
    merged.config.use_cache = True

    out = Path(merged_output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[MERGE] Saving merged model to '{out}' (HF format, safetensors) ...")
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print("[MERGE] Done.")

def save_best_adapter_hf(trainer, tokenizer, out_dir: str):
    """
    Saves the *current* PEFT model (which is already loaded with best weights if
    load_best_model_at_end=True) as Hugging Face LoRA adapter (safetensors) + tokenizer.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ADAPTER] Saving best LoRA adapter to '{out}' (HF format, safetensors) ...")
    trainer.model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    print("[ADAPTER] Done.")

def pick_best_checkpoint_dir(trainer, fallback_dir: str):
    """
    Resolve the best checkpoint directory recorded by Trainer; if unavailable, fallback.
    """
    best = getattr(trainer.state, "best_model_checkpoint", None)
    if best and Path(best).exists():
        return best
    return fallback_dir

# ---- Main ----

def main(args):
    os.environ["WANDB_PROJECT"] = args.wandb_project
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer, model = load_tokenizer_and_model(args)

    print("Loading OpenCoder datasets...")
    raw_educational = load_dataset("OpenCoder-LLM/opc-sft-stage2", "educational_instruct")
    raw_evol        = load_dataset("OpenCoder-LLM/opc-sft-stage2", "evol_instruct")
    raw_mceval      = load_dataset("OpenCoder-LLM/opc-sft-stage2", "mceval_instruct")
    raw_package     = load_dataset("OpenCoder-LLM/opc-sft-stage2", "package_instruct")

    trains, vals = [], {}

    for name, raw, proc in [
        ("educational", raw_educational, process_opencoder_educational),
        ("evol",        raw_evol,        process_opencoder_evol),
        ("mceval",      raw_mceval,      process_opencoder_mceval),
        ("package",     raw_package,     process_opencoder_package),
    ]:
        if "validation" in raw:
            val = raw["validation"].shuffle(seed=42).select(range(min(200, len(raw["validation"]))))
            tr  = raw["train"]
        else:
            train_size = len(raw["train"])
            val_size   = min(200, max(1, train_size // 10))
            split = raw["train"].train_test_split(test_size=val_size, seed=42)
            tr, val = split["train"], split["test"]

        if hasattr(args, 'max_train_samples') and args.max_train_samples > 0:
            tr = tr.select(range(min(args.max_train_samples, len(tr))))

        processed_train = tr.map(lambda x: proc(x, tokenizer, args),
                                 remove_columns=tr.column_names,
                                 desc=f"Processing {name} train")
        processed_val   = val.map(lambda x: proc(x, tokenizer, args),
                                 remove_columns=val.column_names,
                                 desc=f"Processing {name} validation")

        trains.append(processed_train)
        vals[name] = processed_val
        print(f"{name.capitalize()}: {len(processed_train)} train, {len(processed_val)} validation samples")

    train_dataset = concatenate_datasets(trains).shuffle(seed=42)

    # --- NEW: global cap after concatenation ---
    if getattr(args, "max_train_total", 0) and args.max_train_total > 0:
        cap = min(args.max_train_total, len(train_dataset))
        train_dataset = train_dataset.select(range(cap))
        print(f"Total training samples (after global cap): {len(train_dataset)}")
    else:
        print(f"Total training samples: {len(train_dataset)}")

    # which metric to monitor
    if args.best_on == "average":
        metric_key = "eval_avg_loss"
    elif args.best_on in {"educational", "evol", "mceval", "package"}:
        metric_key = f"eval_{args.best_on}_loss"
    else:
        metric_key = args.metric_for_best_model or "eval_loss"

    eff_micro = max(1, min(args.micro_batch_size, 2))  # keep it tiny
    grad_accum = max(1, args.batch_size // eff_micro)

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=eff_micro,
        per_device_eval_batch_size=eff_micro,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",          # <-- fixed (was eval_strategy)
        eval_steps=100,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,                   # keep enough to retain best safely
        load_best_model_at_end=True,
        metric_for_best_model=metric_key,
        greater_is_better=False,
        report_to="wandb",
        run_name=args.run_name,
        dataloader_drop_last=True,
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_torch",  # simple, since only LoRA params have grads
        gradient_checkpointing=True,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        save_safetensors=True,               # ensure HF safetensors for checkpoints
    )

    callbacks = [AverageLossCallback(), EarlyStoppingCallback(early_stopping_patience=5)]

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=vals,  # dict → per-split eval losses
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, padding=True, return_tensors="pt"
        ),
        callbacks=callbacks
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Resolve where the best weights live (trainer already reloaded best into memory)
    best_ckpt_dir = pick_best_checkpoint_dir(trainer, args.output_dir)
    print(f"[BEST] Best checkpoint directory: {best_ckpt_dir}")

    # Save best adapters (HF format) if requested OR needed for merge
    adapters_out = args.adapters_output_dir or (str(Path(args.output_dir).with_name(Path(args.output_dir).name + "-adapters-best")))
    must_save_adapters = args.save_adapters or args.also_save_merged or args.save_merged_only

    if must_save_adapters:
        # Save the CURRENT trainer.model (already best) → guarantees we merge the true best adapter
        save_best_adapter_hf(trainer, tokenizer, adapters_out)

    # Merge and save clean checkpoint
    if args.save_merged_only or args.also_save_merged:
        base_id = args.model_path or args.base_model
        if base_id is None:
            raise ValueError("Need --base_model or --model_path to merge LoRA into base.")
        merged_out = args.merged_output_dir or (str(Path(args.output_dir).with_name(Path(args.output_dir).name + "-merged")))
        merge_and_save_adapters(
            adapters_dir=adapters_out,        # <-- merge the best adapter we just saved
            base_model_id=base_id,
            tokenizer=tokenizer,
            merged_output_dir=merged_out,
            merge_dtype=args.merge_dtype,
            merge_on_gpu=args.merge_on_gpu,
        )

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Simple PEFT-LoRA SFT on OpenCoder, save best adapter (HF) + merged (no FlashAttn2)")
    p.add_argument("--base_model", type=str, default=None, help="HF base model (e.g., meta-llama/Llama-3-8B)")
    p.add_argument("--model_path", type=str, default=None, help="Path to checkpoint to finetune")
    p.add_argument("--output_dir", type=str, default="./sft-opencoder-lora", help="Training output (checkpoints, logs)")
    p.add_argument("--batch_size", type=int, default=256, help="Global batch via grad accumulation")
    p.add_argument("--micro_batch_size", type=int, default=8, help="Per-device micro-batch (clamped to 1–2)")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--cutoff_len", type=int, default=512)
    p.add_argument("--max_train_samples", type=int, default=0, help="Cap per-split train samples before concat (0 = use all)")
    p.add_argument("--max_train_total", type=int, default=0, help="Cap total training examples after concat (0 = no cap). Default 40 for quick test")
    p.add_argument("--train_on_inputs", action="store_true")
    p.add_argument("--add_eos_token", action="store_true")
    p.add_argument("--run_name", type=str, default="opencoder-sft-lora", help="WandB run name for experiment")
    p.add_argument("--wandb_project", type=str, default="opencoder-fine-tuning")
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--system_prompt", type=str, default="You are a helpful coding AI assistant. Provide accurate and well-structured code solutions.")

    # monitoring
    p.add_argument("--best_on", type=str, default="average",
                   choices=["average", "educational", "evol", "mceval", "package", "manual"],
                   help="Monitor per-split, average across splits, or 'manual' if --metric_for_best_model is used.")
    p.add_argument("--metric_for_best_model", type=str, default=None,
                   help="Optional exact HF key like eval_loss or eval_educational_loss; sets best_on=manual.")

    # saving behavior
    p.add_argument("--save_adapters", action="store_true", help="Save best LoRA adapters (HF format) to adapters_output_dir")
    p.add_argument("--save_merged_only", action="store_true", help="Save merged base+LoRA only (no adapters)")
    p.add_argument("--also_save_merged", action="store_true", help="Save merged in addition to adapters")
    p.add_argument("--adapters_output_dir", type=str, default=None, help="Dir for best adapter (defaults to output_dir + '-adapters-best')")
    p.add_argument("--merged_output_dir", type=str, default=None, help="Dir for merged model (defaults to output_dir + '-merged')")
    p.add_argument("--merge_dtype", type=str, default="float16", choices=["float16", "bf16", "float32"], help="Dtype for merged weights")
    p.add_argument("--merge_on_gpu", action="store_true", help="Merge on GPU (uses VRAM); default merges on CPU")

    args = p.parse_args()

    if args.metric_for_best_model:
        args.metric_for_best_model = args.metric_for_best_model.replace("/", "_")
        args.best_on = "manual"

    # Default: if user didn't pick any saving flag, save both best adapters (HF) and merged model
    if not (args.save_adapters or args.also_save_merged or args.save_merged_only):
        args.save_adapters = True
        args.also_save_merged = True

    main(args)
