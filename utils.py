#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def setup_logging(
    logs_dir: str = "logs",
    run_name: str = "tiny_gpt",
    log_file: Optional[str] = None,
) -> Tuple[logging.Logger, Path]:
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    log_path = logs_path / (log_file or f"{run_name}.log")

    logger = logging.getLogger("tiny_gpt")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc

    gc.collect()


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def sequence_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.transpose(1, 2), targets)


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


def save_json(path: str, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    tokenizer: Optional[Any] = None,
    args: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": name or checkpoint_path.stem,
        "checkpoint_name": checkpoint_path.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_state_dict": model.state_dict(),
        "model_config": getattr(model, "config", None),
        "args": args or {},
        "epoch": epoch,
        "loss": loss,
        "metadata": metadata or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if tokenizer is not None:
        payload["tokenizer"] = tokenizer.to_dict()

    torch.save(payload, checkpoint_path)


def plot_loss_curve(history: List[Dict[str, float]], output_path: str, title: str = "Training Loss") -> None:
    """Save a PNG line plot of train/val loss vs. epoch for the README results section."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train_loss"] for entry in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, train_losses, label="train loss")
    if all("val_loss" in entry for entry in history):
        val_losses = [entry["val_loss"] for entry in history]
        plt.plot(epochs, val_losses, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(destination, dpi=150)
    plt.close()


def load_checkpoint(path: str, map_location: Optional[torch.device] = None) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@torch.no_grad()
def sample_gpt(
    model: torch.nn.Module,
    block_size: int,
    stoi: Mapping[str, int],
    itos: Mapping[int, str],
    device: Optional[torch.device] = None,
    start_text: str = "ROMEO:",
    max_new_tokens: int = 400,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = device or next(model.parameters()).device
    model.eval()
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)

    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], dtype=torch.long, device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)

    output = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)[:, -1, :] / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        output.append(itos[int(ix.item())])
        context = torch.cat([context[:, 1:], ix], dim=1)

    return "".join(output)
