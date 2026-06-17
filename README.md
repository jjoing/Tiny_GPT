# TinyGPT: A GPT-2-Style Character-Level Language Model (ECO4126)

A from-scratch, decoder-only Transformer language model trained on classic
literature, built to demonstrate the core architectural ideas behind GPT-2
(token + positional embeddings, masked multi-head self-attention,
feed-forward blocks, residual connections, and layer normalization) at a
scale that trains in minutes rather than days.

> **Honesty note for the oral exam:** this is *not* the pretrained 117M-parameter
> GPT-2 released by OpenAI. It is a small model that follows the same
> architectural blueprint (a stack of pre-norm Transformer decoder blocks
> with causal self-attention) but is trained from random initialization on a
> much smaller, character-level vocabulary. See
> [Differences from official GPT-2](#differences-from-official-gpt-2) for the
> precise list of simplifications and why they were made.

## Project Overview

| File | Description |
| --- | --- |
| `data.py` | text loading, Project Gutenberg boilerplate stripping, built-in dataset downloads, character tokenizer, dataset, dataloaders |
| `model.py` | self-attention blocks and `TinyGPT` |
| `train.py` | training entrypoint, metrics history, loss-curve plotting, checkpoint saving |
| `train_all.py` | convenience runner that trains on all four custom novels in one command |
| `eval.py` | checkpoint loading, loss evaluation, text generation |
| `utils.py` | seed setup, device selection, checkpoint helpers, sampling, loss-curve plotting |

## Architecture

`TinyGPT` ([model.py](model.py)) is a decoder-only Transformer, the same
family of model as GPT-2:

```
tokens ──► token embedding ──┐
                              ├─► + ──► [ Block × num_layers ] ──► LayerNorm ──► Linear (lm_head) ──► next-token logits
positions ──► position embedding ──┘
```

Each `Block` ([model.py:64-75](model.py#L64-L75)) is a **pre-norm Transformer
decoder block**, GPT-2's signature ordering (LayerNorm *before* the
sub-layer, not after, unlike the original 2017 Transformer):

```python
x = x + self.sa(self.ln1(x))      # masked multi-head self-attention, residual
x = x + self.ffwd(self.ln2(x))    # position-wise feed-forward, residual
```

Inside self-attention, each `Head` ([model.py:11-29](model.py#L11-L29))
computes scaled dot-product attention with a causal mask:

```python
weights = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)        # scaled dot product
weights = weights.masked_fill(self.tril[:T, :T] == 0, -inf)     # causal mask
weights = F.softmax(weights, dim=-1)                            # attention distribution
return weights @ v                                               # weighted sum of values
```

`MultiHeadAttention` runs several `Head`s in parallel and concatenates their
outputs, then projects back to `emb_dim` — exactly GPT-2's multi-head
mechanism, just implemented as a loop over small heads instead of one fused
matrix multiply.

## Differences from official GPT-2

| Aspect | This implementation | Official GPT-2 |
| --- | --- | --- |
| Tokenizer | Character-level (`CharTokenizer` in [data.py](data.py)) | Byte-pair encoding (BPE), ~50k subword vocab |
| Weights | Random init, trained from scratch on one book at a time | Pretrained on WebText (40GB) |
| Activation | ReLU in the feed-forward block | GELU |
| Output head | Separate `lm_head` linear layer | Tied with the input token embedding |
| Scale | ~0.5–0.9M parameters, 4 layers | 117M–1.5B parameters, 12–48 layers |
| Positional encoding | Learned absolute position embedding (same family as GPT-2) | Learned absolute position embedding |

These simplifications keep training time and resource needs compatible with
a laptop CPU while preserving every architectural concept the assignment
asks you to demonstrate.

## Custom Dataset

The model trains on **four public-domain novels from Project Gutenberg**,
stored as plain text in [`datasets/`](datasets/):

| Dataset | Author | Raw characters | After cleanup | Vocabulary size |
| --- | --- | --- | --- | --- |
| `alice_in_wonderland.txt` | Lewis Carroll | 144,696 | 144,599 | 75 |
| `frankenstein.txt` | Mary Shelley | 419,434 | 419,336 | 83 |
| `pride_and_prejudice.txt` | Jane Austen | 728,846 | 728,713 | 91 |
| `sherlock_holmes.txt` | Arthur Conan Doyle | 581,425 | 562,202 | 88 |
| `tiny_shakespeare.txt` *(built-in baseline, not used for the assignment results below)* | — | 1,115,394 | 1,115,393 | 65 |

Each Gutenberg `.txt` file ships with a license header and footer wrapped in
`*** START/END OF THE PROJECT GUTENBERG EBOOK ... ***` markers. Left in, this
boilerplate pollutes the character vocabulary and wastes training signal on
legal text. `strip_gutenberg_boilerplate()` in
[data.py](data.py#L40-L55) finds these markers with a regex (handling the
case where the marker itself wraps onto a second line, as in
`sherlock_holmes.txt`) and keeps only the text between them. This single
function is applied uniformly by `read_text()`, so it's a no-op on files
that don't have Gutenberg markers (like `tiny_shakespeare.txt`).

Each model is trained **separately on a single book**, so the model learns
that book's vocabulary and style rather than an averaged mix.

## Environment Setup

```bash
conda create -n TinyGPTenv python=3.10 -y
conda activate TinyGPTenv
pip install -r requirements.txt
```

## Data Pipeline

1. **`read_text()`** ([data.py:164-178](data.py#L164-L178)) loads the raw
   `.txt` file and strips Gutenberg boilerplate.
2. **`CharTokenizer.from_text()`** ([data.py:93-98](data.py#L93-L98)) builds
   the vocabulary directly from the cleaned text: every unique character
   (letters, punctuation, whitespace, even the curly quotes Gutenberg uses)
   becomes one token. `stoi`/`itos` give the int↔char mappings.
3. **`encode_text()`** turns the whole book into a single 1D `torch.long`
   tensor of token ids.
4. **`NextTokenDataset`** ([data.py:125-140](data.py#L125-L140)) exposes a
   sliding window over that tensor: for index `i`, `x = tokens[i : i+block_size]`
   and `y = tokens[i+1 : i+block_size+1]` — i.e. "predict the next character
   at every position in the window." This is the standard self-supervised
   next-token-prediction setup used to train GPT-style models.
5. **`make_data_loaders()`** ([data.py:206-243](data.py#L206-L243)) splits
   the token tensor into a train/val region (`--train-fraction`, default
   `0.9`) and wraps each half in a shuffled/unshuffled `DataLoader`.

## Hyperparameters

| Parameter | Recommended (GPU) | CPU demo (used for the results below) |
| --- | --- | --- |
| `--block-size` (context length) | 128 | 64 |
| `--batch-size` | 64 | 64 |
| `--emb-dim` | 128 | 96 |
| `--num-heads` | 4 | 4 |
| `--num-layers` | 4 | 4 |
| `--dropout` | 0.1 | 0.1 |
| `--epochs` | 20 | 10 |
| `--max-steps` (per epoch) | 200 | 100 |
| `--lr` | 3e-4 | 3e-4 |
| `--train-fraction` | 0.9 | 0.9 |

This repo has no local CUDA GPU, so the results below were generated with
the smaller "CPU demo" column (~6–7 minutes/book on a laptop CPU). The
"Recommended (GPU)" column is what `train_all.py` uses by default — on a
free Colab GPU each book trains in well under a minute and produces
noticeably more fluent samples. Any flag can be overridden; flags passed to
`train_all.py` are forwarded to every run, e.g.:

```bash
python train_all.py --device cuda
```

## Train

Train on all four custom novels in one command (recommended GPU config by
default):

```bash
python train_all.py
```

Override hyperparameters or device for every run (this is how the CPU demo
results below were produced):

```bash
python train_all.py \
    --epochs 10 --max-steps 100 \
    --block-size 64 --emb-dim 96 \
    --device cpu
```

Or train on a single book directly:

```bash
python train.py \
    --dataset sherlock_holmes \
    --run-name sherlock_holmes \
    --epochs 20 --max-steps 200 \
    --block-size 128 --batch-size 64 \
    --emb-dim 128 --num-heads 4 --num-layers 4 \
    --lr 3e-4 --train-fraction 0.9 \
    --data-dir datasets --results-dir results --logs-dir logs
```

To use a brand-new text file instead of one of the four built-in books:

```bash
python train.py --data-path path/to/your_text.txt --run-name my_run
```

Each run writes to `results/<run-name>*`:

```text
results/<run-name>.pt              # checkpoint (weights, tokenizer, config, metrics)
results/<run-name>_history.json    # per-epoch train/val loss
results/<run-name>_loss.png        # loss curve plot
results/<run-name>_sample.txt      # 500-token generated sample at the end of training
logs/<run-name>.log                # full training log
```

Enable Weights & Biases logging:

```bash
python train.py --wandb --wandb-project tiny-gpt
```

For a quick CPU sanity check (a handful of steps, not meant to produce
good samples):

```bash
python train.py --dataset alice_in_wonderland --epochs 1 --max-steps 5 --device cpu
```

## Evaluate and Generate

Evaluate loss and generate text from a saved checkpoint:

```bash
python eval.py \
    --checkpoint results/sherlock_holmes.pt \
    --start-text "The" \
    --max-new-tokens 500
```

Generate only, without computing dataset loss:

```bash
python eval.py \
    --checkpoint results/sherlock_holmes.pt \
    --skip-loss \
    --start-text "The" \
    --max-new-tokens 500
```

Sampling controls:

```bash
python eval.py \
    --checkpoint results/sherlock_holmes.pt \
    --temperature 0.8 \
    --top-k 20
```

## Training Results

<!-- RESULTS_PLACEHOLDER -->

## Notes

- The model is character-level, so each book gets its own vocabulary built
  directly from its (cleaned) text — vocab size is not a fixed hyperparameter.
- Checkpoints store the run name, dataset metadata, model config, tokenizer,
  and final metrics, so `eval.py` can reconstruct the exact model.
- CUDA is used automatically when available (`--device auto`, the default).
  Use `--device cpu` to force CPU, `--device cuda` to require a GPU.
