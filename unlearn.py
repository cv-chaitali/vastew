"""This code is for Big models"""

import argparse
import os
import json
import random
import numpy as np
import torch
import wandb
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import gc


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seeds(1234)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def clear_cache():
    """Clear GPU cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_marginal_loss(operation, batch_correct, batch_incorrect, model, device):
    """Compute marginal loss: difference between incorrect and correct losses."""
    assert operation in ["ga", "gd"], "Operation must be 'ga' or 'gd'."
    
    with torch.cuda.amp.autocast():  # Mixed precision
        correct_loss = get_answer_loss("gd", batch_correct, model, device)
        incorrect_loss = get_answer_loss("gd", batch_incorrect, model, device)

        # if either is constant w/o grad, the diff will also be constant -> we'll check requires_grad upstream
        marginal = incorrect_loss - correct_loss
        if operation == "ga":
            return marginal
        else:
            return -marginal


def get_answer_loss(operation, batch, model, device):
    """Compute GA (maximize) or GD (minimize) per-token CE -> mean."""
    assert operation in ["ga", "gd"], "Operation must be 'ga' or 'gd'."
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    with torch.cuda.amp.autocast():  # Mixed precision
        outputs = model(input_ids, attention_mask=attention_mask)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        batch_size, seq_len = shift_labels.shape

        mask = shift_labels != -100
        per_token_loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
        ).reshape(batch_size, seq_len)

        if operation == "ga":
            per_token_loss = -per_token_loss

        # average only over valid label positions, per example; then mean over batch
        valid_means = []
        for i in range(batch_size):
            valid = mask[i]
            if valid.any():
                valid_means.append(per_token_loss[i][valid].mean())

        if len(valid_means) == 0:
            # no valid labels -> return a constant zero (no grad)
            return torch.tensor(0.0, device=device)

        return torch.stack(valid_means).mean()


def build_tokens(tokenizer, system, user, assistant, args):
    """Build tokenized conversation; respect chat_template when available."""
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        if assistant:
            msgs.append({"role": "assistant", "content": assistant})
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=(assistant is None)
        )
    else:
        parts = []
        if system:
            parts.append(f"System: {system}")
        parts.append(f"User: {user}")
        if assistant:
            parts.append(f"Assistant: {assistant}")
            text = "\n".join(parts)
        else:
            text = "\n".join(parts) + "\nAssistant:"

    toks = tokenizer(
        text, truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )

    if assistant is not None:
        labels = toks["input_ids"].clone()
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            a_start = text.find(assistant)
        else:
            a_start = text.find(f"Assistant: {assistant}")
            if a_start != -1:
                a_start = text.find(assistant, a_start)
        if a_start != -1:
            prefix = text[:a_start]
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            labels[:, :len(prefix_ids)] = -100
        labels[labels == tokenizer.pad_token_id] = -100
    else:
        labels = torch.full_like(toks["input_ids"], -100)

    return {
        "input_ids": toks["input_ids"].squeeze(0),
        "attention_mask": toks["attention_mask"].squeeze(0),
        "labels": labels.squeeze(0)
    }


# ---------- OpenCoder dataset ----------
def process_opcoder(dp, tokenizer, args):
    """
    Build prompt from OpenCoder-LLM/opc-sft-stage2 (educational_instruct).
    Fields: instruction, output, code?, testcase?, entrypoint?
    """
    system = args.system_prompt
    instruction = (dp.get("instruction") or "").strip()
    output = (dp.get("output") or "").strip()
    code = (dp.get("code") or "")
    testcase = (dp.get("testcase") or "")
    entrypoint = (dp.get("entrypoint") or "")

    user_parts = [f"Instruction:\n{instruction}"]
    if entrypoint:
        user_parts.append(f"\nEntrypoint:\n{entrypoint}")
    if testcase:
        user_parts.append(f"\nTestcases:\n{testcase}")
    if code:
        user_parts.append(f"\nCode Context:\n{code}")
    user_parts.append("\nProvide your final answer below.")
    user = "\n".join(user_parts)

    assistant = output if output else ""
    return build_tokens(tokenizer, system, user, assistant, args)


# ---------- Synthetic unlearning (streamed JSON/JSONL) ----------
def create_incorrect_answer(question, tokenizer, args):
    bads = [
        "I know the answer to this question.",
        "This is clearly explained in my training data.",
        "The answer is well documented.",
        "I can provide detailed information about this.",
        "This falls within my knowledge base."
    ]
    return build_tokens(
        tokenizer, args.system_prompt, question, random.choice(bads), args
    )


def preprocess_synthetic_marginal(example, tokenizer, args):
    question = example.get("question", "")
    correct_tokens = build_tokens(
        tokenizer, args.system_prompt, question, "I don't know.", args
    )
    incorrect_tokens = create_incorrect_answer(question, tokenizer, args)
    return {"correct": correct_tokens, "incorrect": incorrect_tokens}


def count_jsonl_lines(path):
    cnt = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            cnt += 1
    return cnt


def iter_jsonl_processed(path, tokenizer, args):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield preprocess_synthetic_marginal(obj, tokenizer, args)


def main():
    parser = argparse.ArgumentParser(
        description="Unlearn synthetic data but retain code instruction-following using marginal loss."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--synthetic_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="unlearned_model")
    parser.add_argument("--batch_size", type=int, default=2)  # Reduced from 8
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=384)  # Reduced from 512
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--marginal_weight", type=float, default=1.0)
    parser.add_argument("--med_weight", type=float, default=1.0)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful coding assistant and a coding expert.")
    parser.add_argument("--samples_per_dataset", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)  # Increased from 4
    parser.add_argument("--hf_cache_dir", type=str, default="/disk2/hf_cache")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit Adam optimizer")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    args = parser.parse_args()

    # route caches to roomy disk
    os.environ.setdefault("HF_HOME", args.hf_cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(args.hf_cache_dir, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(args.hf_cache_dir, "models"))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args_unlearning.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb.init(project="unlearning-llms", config=vars(args),
               name=args.run_name, reinit=True)

    # Load model with memory optimizations
    print("Loading model with memory optimizations...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, 
        device_map='auto',
        torch_dtype=torch.float16,  # Use half precision
        low_cpu_mem_usage=True,     # Reduce CPU memory usage during loading
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory efficiency
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # synthetic loader (streamed)
    synth_path = args.synthetic_dataset_path
    if not os.path.exists(synth_path):
        raise FileNotFoundError(f"Synthetic dataset not found at: {synth_path}")
    ext = synth_path.split(".")[-1].lower()
    if ext not in ("jsonl", "json"):
        raise ValueError("Synthetic data must be .jsonl or .json")

    if ext == "json":
        with open(synth_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        processed_synth_mem = [preprocess_synthetic_marginal(ex, tokenizer, args) for ex in raw]
        num_synth_examples = len(processed_synth_mem)

        def new_synth_iter():
            for item in processed_synth_mem:
                yield item
    else:
        num_synth_examples = count_jsonl_lines(synth_path)

        def new_synth_iter():
            return iter_jsonl_processed(synth_path, tokenizer, args)

    # OpenCoder dataset
    print("Loading OpenCoder-LLM/opc-sft-stage2 (educational_instruct)...")
    raw_opc = load_dataset(
        "OpenCoder-LLM/opc-sft-stage2", "educational_instruct", split="train",
        cache_dir=os.environ.get("HF_DATASETS_CACHE", None)
    )
    ds_opc = raw_opc.select(range(min(args.samples_per_dataset, len(raw_opc))))
    ds_opc = ds_opc.map(lambda ex: process_opcoder(ex, tokenizer, args),
                        remove_columns=ds_opc.column_names)
    ds_opc.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Use pin_memory=False and num_workers=0 to reduce memory overhead
    opc_loader = DataLoader(ds_opc, batch_size=args.batch_size, shuffle=True, 
                           pin_memory=False, num_workers=0)
    print(f"OpenCoder: {len(ds_opc)} samples")

    # Use 8-bit optimizer if requested
    if args.use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print("Using 8-bit AdamW optimizer")
            # Don't use gradient scaler with 8-bit optimizer
            use_scaler = False
        except ImportError:
            print("bitsandbytes not available, falling back to regular AdamW")
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            use_scaler = True
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        use_scaler = True
    
    total_steps = max(num_synth_examples, len(opc_loader)) * args.num_train_epochs // max(1, args.gradient_accumulation_steps)
    scheduler = get_scheduler(
        args.scheduler, optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=max(1, total_steps)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Initialize gradient scaler only if not using 8-bit optimizer
    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()
        print("Using gradient scaler with mixed precision")
    else:
        scaler = None
        print("Skipping gradient scaler (incompatible with 8-bit optimizer)")

    global_step = 0
    print(f"Starting training for {args.num_train_epochs} epochs...")

    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        synth_iter = new_synth_iter()
        opc_iter = iter(opc_loader)
        max_steps = max(num_synth_examples, len(opc_loader))

        for step in tqdm(range(max_steps), desc=f"Epoch {epoch+1}"):
            # accumulate log-only loss as python float
            accumulated_loss_val = 0.0
            loss_components = {}

            for micro_step in range(args.gradient_accumulation_steps):
                # Build a loss that only backprops if it has a graph
                micro_loss = None  # None means "no contribution this micro-step"

                # --- synthetic (marginal) ---
                try:
                    synth_data = next(synth_iter)
                    batch_correct = {k: v.unsqueeze(0).to(device) for k, v in synth_data["correct"].items()}
                    batch_incorrect = {k: v.unsqueeze(0).to(device) for k, v in synth_data["incorrect"].items()}
                    marginal_loss = get_marginal_loss("ga", batch_correct, batch_incorrect, model, device)
                    if marginal_loss.requires_grad:
                        micro_loss = (marginal_loss * args.marginal_weight) if micro_loss is None else (micro_loss + args.marginal_weight * marginal_loss)
                        loss_components["marginal_loss"] = float(marginal_loss.detach().item())
                except StopIteration:
                    pass

                # --- OpenCoder (GD) ---
                try:
                    batch_o = next(opc_iter)
                except StopIteration:
                    opc_iter = iter(opc_loader)
                    try:
                        batch_o = next(opc_iter)
                    except StopIteration:
                        batch_o = None

                if batch_o is not None:
                    batch_o = {k: v.to(device) for k, v in batch_o.items()}
                    if (batch_o["labels"] != -100).any():
                        gd_o = get_answer_loss("gd", batch_o, model, device)
                        if gd_o.requires_grad:
                            micro_loss = (gd_o * args.med_weight) if micro_loss is None else (micro_loss + args.med_weight * gd_o)
                            loss_components["opc_gd"] = float(gd_o.detach().item())

                # If nothing contributed this micro-step, skip
                if micro_loss is None:
                    continue

                micro_loss = micro_loss / args.gradient_accumulation_steps
                accumulated_loss_val += float(micro_loss.detach().item())
                
                # Handle backward pass with or without scaler
                if use_scaler and scaler is not None:
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()

            # Optimizer step with proper gradient handling
            try:
                if use_scaler and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Clip gradients manually if not using scaler
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
            except RuntimeError as e:
                if "unscale FP16 gradients" in str(e):
                    print(f"Gradient scaler error at step {step}, skipping step: {e}")
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            # Clear cache every few steps to prevent memory buildup
            if step % 10 == 0:
                clear_cache()

            logs = {
                "total_loss": accumulated_loss_val,
                "step": global_step,
                "epoch": epoch + 1
            }
            logs.update(loss_components)
            wandb.log(logs)

        # Clear cache at the end of each epoch
        clear_cache()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
