#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Optional, Tuple

import torch

from data import (
    DATASETS,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_NAME,
    CharTokenizer,
    create_dataloader,
    read_text,
)
from model import TinyGPT
from utils import get_device, load_checkpoint, sample_gpt, sequence_cross_entropy


@torch.no_grad()
def evaluate_loss(
    model: TinyGPT,
    loader,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    if total_count == 0:
        raise ValueError("loader did not yield any batches")
    return total_loss / total_count


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[TinyGPT, CharTokenizer, dict]:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model_config = checkpoint["model_config"]
    tokenizer = CharTokenizer.from_dict(checkpoint["tokenizer"])

    model = TinyGPT.from_config(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate or sample from a TinyGPT checkpoint.")
    parser.add_argument("--checkpoint", default="results/tiny_gpt.pt")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=sorted(DATASETS),
        help="Dataset for loss evaluation; defaults to checkpoint metadata.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-loss", action="store_true")
    parser.add_argument("--start-text", default="The")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    model, tokenizer, checkpoint = load_model_from_checkpoint(args.checkpoint, device)
    block_size = checkpoint["model_config"]["block_size"]

    if not args.skip_loss:
        metadata = checkpoint.get("metadata", {})
        dataset = args.dataset or metadata.get("dataset", DEFAULT_DATASET_NAME)
        data_path = args.data_path or metadata.get("data_path")
        text = read_text(
            path=data_path,
            download=not args.no_download,
            dataset=dataset,
            data_dir=args.data_dir,
        )
        loader, _ = create_dataloader(
            text=text,
            tokenizer=tokenizer,
            block_size=block_size,
            batch_size=args.batch_size,
            shuffle=False,
        )
        loss = evaluate_loss(model, loader, device, max_steps=args.max_steps)
        print(f"loss {loss:.4f}")

    sample = sample_gpt(
        model=model,
        block_size=block_size,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        device=device,
        start_text=args.start_text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(sample)


if __name__ == "__main__":
    main()
