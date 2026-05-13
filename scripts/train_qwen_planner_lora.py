from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError as exc:  # pragma: no cover - runtime setup guard
    raise SystemExit("Missing dependency: pip install peft==0.14.0") from exc


DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class ChatSftDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_length: int) -> None:
        self.items: List[Dict[str, List[int]]] = []
        for row in rows:
            messages = row["messages"]
            prompt_messages = messages[:-1]
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            full = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False)
            prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, add_special_tokens=False)
            input_ids = list(full["input_ids"])
            attention_mask = list(full["attention_mask"])
            labels = list(input_ids)
            prompt_len = min(len(prompt["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            if all(x == -100 for x in labels):
                continue
            self.items.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.items[idx]


def collate_batch(batch: List[Dict[str, List[int]]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for item in batch:
        pad = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(item["attention_mask"] + [0] * pad)
        labels.append(item["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses: List[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = read_jsonl(args.train_jsonl)
    eval_rows = read_jsonl(args.eval_jsonl) if args.eval_jsonl and args.eval_jsonl.exists() else []
    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples:
        eval_rows = eval_rows[: args.max_eval_samples]

    train_data = ChatSftDataset(train_rows, tokenizer, args.max_length)
    eval_data = ChatSftDataset(eval_rows, tokenizer, args.max_length) if eval_rows else None
    if not train_data:
        raise SystemExit("No train examples survived tokenization.")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )
    eval_loader = (
        DataLoader(
            eval_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )
        if eval_data
        else None
    )

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn,
        trust_remote_code=True,
    ).to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_update_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_update_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_path": str(args.model_path),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": target_modules,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_length": args.max_length,
        "attn": args.attn,
    }
    (args.output_dir / "train_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "loss": round(float(loss.detach().cpu()) * args.grad_accum, 6),
                                "lr": scheduler.get_last_lr()[0],
                                "elapsed_sec": round(elapsed, 2),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                if eval_loader and global_step % args.eval_every == 0:
                    eval_loss = evaluate(model, eval_loader, device)
                    print(json.dumps({"global_step": global_step, "eval_loss": round(eval_loss, 6)}, ensure_ascii=False), flush=True)

    if eval_loader:
        eval_loss = evaluate(model, eval_loader, device)
        print(json.dumps({"final_eval_loss": round(eval_loss, 6)}, ensure_ascii=False), flush=True)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(json.dumps({"saved": str(args.output_dir), "global_step": global_step}, ensure_ascii=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Qwen planner LoRA from chat JSONL.")
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/ARACHNE-X/weights/Qwen3-4B-Instruct-2507"))
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--eval-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=str, default=DEFAULT_TARGET_MODULES)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--attn", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260513)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
