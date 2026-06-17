#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from data import (
    DATASETS,
    DEFAULT_DATASET_DIR,
    DEFAULT_DATASET_NAME,
    make_data_loaders,
    read_text,
    resolve_data_path,
)
from eval import evaluate_loss
from model import TinyGPT
from utils import (
    clear_cache,
    count_parameters,
    ensure_dir,
    get_device,
    plot_loss_curve,
    sample_gpt,
    save_checkpoint,
    save_json,
    sequence_cross_entropy,
    set_seed,
    setup_logging,
)


def train_one_epoch(
    model: TinyGPT,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for step, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    if total_count == 0:
        raise ValueError("loader did not yield any batches")
    return total_loss / total_count


def safe_run_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    return safe.strip("._-") or "tiny_gpt"


def make_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return safe_run_name(args.run_name)

    dataset_label = args.dataset if args.data_path is None else Path(args.data_path).stem
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return safe_run_name(f"tiny_gpt-{dataset_label}-{timestamp}")


def init_wandb(args: argparse.Namespace, run_name: str, config: dict):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed; run pip install -r requirements.txt") from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the notebook TinyGPT model.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        choices=sorted(DATASETS),
        help="Built-in dataset to use when --data-path is not provided.",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--data-path", default=None, help="Optional custom text file path.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Defaults to '<run-name>.pt' so multiple runs don't overwrite each other.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="tiny-gpt")
    parser.add_argument("--wandb-entity", default=None)

    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--start-text", default="The")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    data_path = resolve_data_path(
        path=args.data_path,
        dataset=args.dataset,
        data_dir=args.data_dir,
    )
    run_name = make_run_name(args)
    checkpoint_name = args.checkpoint_name or f"{run_name}.pt"
    checkpoint_path = Path(args.results_dir) / checkpoint_name
    logger, log_path = setup_logging(
        logs_dir=args.logs_dir,
        run_name=run_name,
        log_file=args.log_file,
    )

    set_seed(args.seed)
    device = get_device(args.device)
    text = read_text(
        path=args.data_path,
        download=not args.no_download,
        dataset=args.dataset,
        data_dir=args.data_dir,
    )
    train_loader, val_loader, tokenizer = make_data_loaders(
        text=text,
        block_size=args.block_size,
        batch_size=args.batch_size,
        train_fraction=args.train_fraction,
    )

    model = TinyGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    parameter_count = count_parameters(model)

    ensure_dir(args.results_dir)

    run_config = vars(args).copy()
    run_config.update(
        {
            "run_name": run_name,
            "data_path": str(data_path),
            "checkpoint_path": str(checkpoint_path),
            "log_path": str(log_path),
            "parameters": parameter_count,
            "vocab_size": tokenizer.vocab_size,
        }
    )
    wandb_run = init_wandb(args, run_name=run_name, config=run_config)

    logger.info("run name: %s", run_name)
    logger.info("dataset: %s", args.dataset)
    logger.info("data path: %s", data_path)
    logger.info("log file: %s", log_path)
    logger.info("device: %s", device)
    logger.info("vocab size: %s", tokenizer.vocab_size)
    logger.info("parameters: %s", f"{parameter_count:,}")

    last_val_loss = None
    history = []
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            max_steps=args.max_steps,
        )

        message = f"epoch {epoch:2d} | train loss {train_loss:.4f}"
        metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "lr": args.lr,
        }
        history_entry = {"epoch": epoch + 1, "train_loss": train_loss}
        if val_loader is not None:
            last_val_loss = evaluate_loss(model, val_loader, device, max_steps=args.max_steps)
            metrics["val/loss"] = last_val_loss
            history_entry["val_loss"] = last_val_loss
            message += f" | val loss {last_val_loss:.4f}"
        logger.info(message)
        history.append(history_entry)

        if wandb_run is not None:
            wandb_run.log(metrics, step=epoch + 1)

        if args.sample_every > 0 and (epoch + 1) % args.sample_every == 0:
            sample = sample_gpt(
                model=model,
                block_size=args.block_size,
                stoi=tokenizer.stoi,
                itos=tokenizer.itos,
                device=device,
                start_text=args.start_text,
                max_new_tokens=args.max_new_tokens,
            )
            logger.info("sample:\n%s", sample)

    final_metrics = {
        "epoch": args.epochs - 1,
        "train_loss": train_loss,
    }
    if last_val_loss is not None:
        final_metrics["val_loss"] = last_val_loss

    history_path = Path(args.results_dir) / f"{run_name}_history.json"
    plot_path = Path(args.results_dir) / f"{run_name}_loss.png"
    sample_path = Path(args.results_dir) / f"{run_name}_sample.txt"

    save_json(str(history_path), {"run_name": run_name, "dataset": args.dataset, "history": history})
    plot_loss_curve(history, str(plot_path), title=f"{run_name} loss curve")

    final_sample = sample_gpt(
        model=model,
        block_size=args.block_size,
        stoi=tokenizer.stoi,
        itos=tokenizer.itos,
        device=device,
        start_text=args.start_text,
        max_new_tokens=max(args.max_new_tokens, 500),
    )
    sample_path.write_text(final_sample, encoding="utf-8")
    logger.info("saved loss history: %s", history_path)
    logger.info("saved loss plot: %s", plot_path)
    logger.info("saved final sample: %s", sample_path)

    metadata = {
        "run_name": run_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "dataset": args.dataset,
        "data_path": str(data_path),
        "logs_dir": args.logs_dir,
        "log_path": str(log_path),
        "history_path": str(history_path),
        "plot_path": str(plot_path),
        "sample_path": str(sample_path),
        "wandb_project": args.wandb_project if args.wandb else None,
    }

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        args=vars(args),
        epoch=args.epochs - 1,
        loss=train_loss,
        name=run_name,
        metadata=metadata,
        metrics=final_metrics,
    )
    clear_cache()
    logger.info("saved checkpoint: %s", checkpoint_path)

    if wandb_run is not None:
        for key, value in final_metrics.items():
            wandb_run.summary[key] = value
        wandb_run.summary["checkpoint_path"] = str(checkpoint_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
