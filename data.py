#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import request

import torch
from torch.utils.data import DataLoader, Dataset

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_DATASET_DIR = "datasets"
DEFAULT_DATASET_NAME = "tiny_shakespeare"


@dataclass(frozen=True)
class TextDatasetSpec:
    name: str
    url: str
    filename: str
    description: str


DATASETS: Dict[str, TextDatasetSpec] = {
    "tiny_shakespeare": TextDatasetSpec(
        name="tiny_shakespeare",
        url=TINY_SHAKESPEARE_URL,
        filename="tiny_shakespeare.txt",
        description="Tiny Shakespeare from Karpathy's char-rnn examples.",
    ),
    "alice_in_wonderland": TextDatasetSpec(
        name="alice_in_wonderland",
        url="https://www.gutenberg.org/files/11/11-0.txt",
        filename="alice_in_wonderland.txt",
        description="Alice's Adventures in Wonderland by Lewis Carroll.",
    ),
    "pride_and_prejudice": TextDatasetSpec(
        name="pride_and_prejudice",
        url="https://www.gutenberg.org/files/1342/1342-0.txt",
        filename="pride_and_prejudice.txt",
        description="Pride and Prejudice by Jane Austen.",
    ),
    "sherlock_holmes": TextDatasetSpec(
        name="sherlock_holmes",
        url="https://www.gutenberg.org/files/1661/1661-0.txt",
        filename="sherlock_holmes.txt",
        description="The Adventures of Sherlock Holmes by Arthur Conan Doyle.",
    ),
    "frankenstein": TextDatasetSpec(
        name="frankenstein",
        url="https://www.gutenberg.org/files/84/84-0.txt",
        filename="frankenstein.txt",
        description="Frankenstein by Mary Shelley.",
    ),
}


_GUTENBERG_START_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_GUTENBERG_END_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
_SCENE_BREAK_RE = re.compile(r"^[ \t]*(?:\*[ \t]*){2,}$\n?", re.MULTILINE)
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_REPEATED_SPACE_RE = re.compile(r"[ \t]{2,}")


def collapse_repeated_spaces(text: str) -> str:
    """Collapse runs of 2+ spaces/tabs into a single space.

    Gutenberg transcriptions use long space runs to center title pages and
    illustration captions and to align table-of-contents page numbers.
    These never occur in actual prose, so at char level they're noise an
    undertrained model latches onto -- the same failure mode as the
    asterisk scene breaks handled by strip_scene_breaks.
    """
    return _REPEATED_SPACE_RE.sub(" ", text)


def strip_scene_breaks(text: str) -> str:
    """Drop typographic scene-break lines (e.g. "*    *    *    *").

    Some Gutenberg transcriptions (Alice in Wonderland in particular) render
    a printed scene divider as a line of repeated asterisks and spaces. At
    char level this is a frequent, low-entropy pattern that an undertrained
    model latches onto, producing generated text full of spaces and "*".
    """
    text = _SCENE_BREAK_RE.sub("", text)
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", text)


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop Project Gutenberg license header/footer, keeping only the book body.

    The four custom novels are sourced from Project Gutenberg, which wraps the
    actual text with a license preamble and postamble delimited by
    "*** START/END OF THE PROJECT GUTENBERG EBOOK ... ***" markers (the
    marker itself sometimes wraps onto a second line, hence DOTALL). Leaving
    this boilerplate in would waste vocabulary and training signal on
    legal text instead of the book's prose.
    """
    start_match = _GUTENBERG_START_RE.search(text)
    end_match = _GUTENBERG_END_RE.search(text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(text)
    return text[start_idx:end_idx].strip()


def get_dataset_spec(name: str) -> TextDatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {name!r}; available: {available}") from exc


def get_dataset_path(
    dataset: str = DEFAULT_DATASET_NAME,
    data_dir: str = DEFAULT_DATASET_DIR,
) -> Path:
    spec = get_dataset_spec(dataset)
    return Path(data_dir) / spec.filename


def resolve_data_path(
    path: Optional[str] = None,
    dataset: str = DEFAULT_DATASET_NAME,
    data_dir: str = DEFAULT_DATASET_DIR,
) -> Path:
    return Path(path) if path else get_dataset_path(dataset=dataset, data_dir=data_dir)


@dataclass
class CharTokenizer:
    chars: List[str]
    stoi: Dict[str, int]
    itos: Dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "CharTokenizer":
        chars = list(payload["chars"])
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(chars=chars, stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> List[int]:
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            preview = "".join(repr(ch) for ch in unknown[:5])
            raise ValueError(f"text contains characters outside the tokenizer: {preview}")
        return [self.stoi[ch] for ch in text]

    def decode(self, indices: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in indices)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"chars": self.chars}


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        if data.dim() != 1:
            raise ValueError("data must be a 1D tensor of token ids")
        if len(data) <= block_size:
            raise ValueError("data length must be larger than block_size")
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


def download_text(path: str, url: str = TINY_SHAKESPEARE_URL) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url_request = request.Request(url, headers={"User-Agent": "TinyGPT dataset downloader"})
    with request.urlopen(url_request) as response:
        destination.write_bytes(response.read())
    return destination


def download_dataset(
    dataset: str = DEFAULT_DATASET_NAME,
    data_dir: str = DEFAULT_DATASET_DIR,
    force: bool = False,
) -> Path:
    spec = get_dataset_spec(dataset)
    destination = get_dataset_path(dataset=dataset, data_dir=data_dir)
    if destination.exists() and not force:
        return destination
    return download_text(str(destination), url=spec.url)


def read_text(
    path: Optional[str] = None,
    download: bool = True,
    dataset: str = DEFAULT_DATASET_NAME,
    data_dir: str = DEFAULT_DATASET_DIR,
) -> str:
    data_path = resolve_data_path(path=path, dataset=dataset, data_dir=data_dir)
    if not data_path.exists():
        if not download:
            raise FileNotFoundError(f"{data_path} does not exist")
        if path is None:
            download_dataset(dataset=dataset, data_dir=data_dir)
        else:
            download_text(str(data_path), url=get_dataset_spec(dataset).url)
    text = strip_gutenberg_boilerplate(data_path.read_text(encoding="utf-8"))
    text = strip_scene_breaks(text)
    return collapse_repeated_spaces(text)


def encode_text(text: str, tokenizer: Optional[CharTokenizer] = None) -> Tuple[torch.Tensor, CharTokenizer]:
    tokenizer = tokenizer or CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return encoded, tokenizer


def create_dataloader(
    text: str,
    tokenizer: Optional[CharTokenizer],
    block_size: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Tuple[DataLoader, CharTokenizer]:
    encoded, tokenizer = encode_text(text, tokenizer)
    dataset = NextTokenDataset(encoded, block_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader, tokenizer


def make_data_loaders(
    text: str,
    block_size: int = 64,
    batch_size: int = 64,
    train_fraction: float = 1.0,
    tokenizer: Optional[CharTokenizer] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], CharTokenizer]:
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")

    encoded, tokenizer = encode_text(text, tokenizer)

    if train_fraction == 1.0:
        train_data = encoded
        val_data = None
    else:
        split_idx = int(len(encoded) * train_fraction)
        split_idx = max(block_size + 1, min(split_idx, len(encoded) - block_size - 1))
        train_data = encoded[:split_idx]
        val_data = encoded[split_idx:]

    train_loader = DataLoader(
        NextTokenDataset(train_data, block_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = None
    if val_data is not None and len(val_data) > block_size:
        val_loader = DataLoader(
            NextTokenDataset(val_data, block_size),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return train_loader, val_loader, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare built-in text datasets.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME,
        choices=sorted(DATASETS) + ["all"],
        help="Dataset to download, or 'all' for every built-in dataset.",
    )
    parser.add_argument(
        "--output-dir",
        "--data-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory used for built-in dataset files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a single dataset; defaults to output-dir/<dataset>.txt.",
    )
    parser.add_argument("--url", default=None, help="Optional URL override for a single dataset.")
    parser.add_argument("--list", action="store_true", help="List built-in datasets and exit.")
    parser.add_argument("--force", action="store_true", help="Overwrite files if they already exist.")
    return parser.parse_args()


def print_dataset_summary(dataset: str, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    print(f"dataset: {dataset}")
    print(f"saved dataset: {path}")
    print(f"characters: {len(text):,}")
    print(f"vocab size: {tokenizer.vocab_size}")


def main() -> None:
    args = parse_args()

    if args.list:
        for name in sorted(DATASETS):
            spec = DATASETS[name]
            print(f"{name:20s} {spec.filename:28s} {spec.description}")
        return

    if args.dataset == "all":
        if args.output is not None or args.url is not None:
            raise SystemExit("--output and --url can only be used with a single dataset")
        for dataset in sorted(DATASETS):
            output_path = download_dataset(
                dataset=dataset,
                data_dir=args.output_dir,
                force=args.force,
            )
            print_dataset_summary(dataset, output_path)
            print()
        return

    spec = get_dataset_spec(args.dataset)
    output_path = Path(args.output) if args.output else get_dataset_path(args.dataset, args.output_dir)

    if output_path.exists() and not args.force:
        print(f"dataset already exists: {output_path}")
    else:
        download_text(str(output_path), url=args.url or spec.url)

    print_dataset_summary(args.dataset, output_path)


if __name__ == "__main__":
    main()
