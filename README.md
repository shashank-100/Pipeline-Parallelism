# Pipeline Parallelism

A PyTorch implementation of pipeline parallelism strategies for training large neural networks across multiple GPUs/devices.

## Overview

This project implements three pipeline parallelism schedules:

- **Naive (GPipe-style)**: Simple stop-and-wait pipeline with all-forward then all-backward
- **GPipe**: Micro-batch pipeline with forward warmup, then backward drain
- **1F1B (PipeDream)**: One-Forward-One-Backward schedule for improved pipeline efficiency

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage

Run with 4 pipeline stages:

```bash
uv run torchrun --nproc-per-node 4 src/main.py
```

## Configuration

Edit `src/config/config.py` to adjust:

- `BATCH_SIZE`: Input batch size
- `HIDDEN_DIM`: Model hidden dimension
- `TOTAL_LAYERS`: Total number of layers (divided across stages)
- `CHUNKS`: Number of micro-batches for GPipe/1F1B
- `STEPS`: Training steps

## Project Structure

```bash
src/
├── main.py           # Main training loop
├── model.py          # ShardedMLP model definition
├── schedule.py       # Pipeline schedules (Naive, GPipe, 1F1B)
├── communication.py  # Distributed communication primitives
├── profiler.py       # Performance profiling utilities
└── config/
    └── config.py     # Training configuration
```

## Pipeline Schedules

### Naive Pipeline

Simple sequential execution: forward all micro-batches, then backward all micro-batches.

### GPipe

Splits batches into micro-batches and pipelines them through stages:

- Forward warmup phase
- Backward drain in reverse order
- Reduces pipeline bubbles compared to naive approach

### 1F1B (One-Forward-One-Backward)

Interleaves forward and backward passes for better efficiency:

- Warmup: Forward-only passes to fill pipeline
- Steady state: Alternating forward and backward
- Drain: Backward-only passes to complete remaining gradients
- Uses async communication to prevent blocking

## Performance Profiling

The profiler tracks:

- Compute time (forward/backward)
- Communication time (send/receive)
- Pipeline bubbles (idle time)

Results are printed after training for each rank.

## License

See [LICENSE](README.md) file for details.

## Credits

thanks to **[freeCodeCamp](https://www.freecodecamp.org/)** , **[kiankyars](https://github.com/kiankyars)** for teaching this, goated stuff.

(I needed to take the help of claude while implementing onef_oneb as i was stuggling with async behaviour)