# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), and one of:
- **NVIDIA GPU** (H100 recommended) — full performance with Flash Attention 3 and `torch.compile`
- **Apple Silicon Mac** (M1/M2/M3/M4) — runs via PyTorch MPS backend (~3–5× slower throughput than H100, but fully functional)

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

### Apple Silicon (MPS) notes

The codebase auto-detects the available device at startup (`cuda → mps → cpu`). On Apple Silicon, the following differences apply automatically — no flags or config changes needed:

| Feature | CUDA (H100) | Apple Silicon (MPS) |
|---|---|---|
| Attention kernel | Flash Attention 3 | `F.scaled_dot_product_attention` |
| `torch.compile` | ✅ Enabled | ❌ Disabled (MPS inductor unsupported) |
| Mixed precision | bfloat16 autocast | fp32 (embeddings stay bfloat16) |
| Memory tracking | `cuda.max_memory_allocated` | Running max of `mps.driver_allocated_memory` |
| MFU reference | H100 BF16 (989.5 TFLOPS) | M4 Max GPU FP16 (~14.2 TFLOPS) |

Because throughput is lower on Apple Silicon, **`val_bpb` results are not directly comparable between CUDA and MPS runs** — each platform should establish its own baseline. The recommended Apple Silicon setup for the experiment loop:

```bash
# Fewer shards to keep prep fast on laptop storage
uv run prepare.py --num-shards 8 --download-workers 4

# Reduce batch size if you hit memory pressure (default is 128)
# Edit DEVICE_BATCH_SIZE in train.py before running
uv run train.py
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos)

## Platform research

See [`mlx-feasibility.md`](mlx-feasibility.md) for a detailed analysis of porting options for Apple Silicon, including a comparison of MLX-native, hybrid, and PyTorch/MPS approaches with pros/cons for each.

## License

MIT
