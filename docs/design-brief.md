# Design Brief

## Goal

Build a distributed GPT trainer from scratch that supports
single-process, multi-process CPU, and multi-GPU training. The project
should prioritize correctness and reproducibility while
providing a UI to visualize training progress and distributed
communication.

## Initial scope
- Data parallelism
- Gradient averaging
- Checkpoint & resume
- Custom ring all-reduce

## Non-goals (for now; this may change)
- Pipeline parallelism
- Tensor parallelism
- Mixture-of-Experts models

## Success criteria
- Data should be across processes successfully and combined
- Training can resume from a valid checkpoint and produce the same subsequent training trajectory as an uninterrupted run
- Benchmarks are reproducible using the documented benchmark protocol and report throughput, latency, and scaling efficiency