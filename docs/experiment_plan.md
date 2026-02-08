# Experirements

Goal is to measure perf of the baseline architecture, then see how much we can improve from the baseline.

Metrics will be:

- Compare at 25, 50, and 100% of compute budget (GPU Hours)
- Compare time to cross threshold of metrics (MSE, MAE)

Things to test:

- **baseline:** TimeMOE with pointwise tokenization
- Patching
- Sliding window attention
- QK normalization
- FP8 matmuls
- Batch size schedule
- Gradient Compression (SquishGrad)

Model size: 50M active?

## Setup

- data mixture + sampling:
  - TimeMOE dataset
  - Fev Evaluation datasets
- context length: 4096 (for variants run at 512)
- horizon distribution: [1, 8, 32, 64]
- train budget: 2 GPU hours (H100)
- metrics: overall + horizon-binned + per-frequency bucket
- 3 fixed seeds for final comparisons (use 1 seed for early triage)

Always log:

- cumulative GPU-seconds
- tokens/time-points processed
- effective sequence length (patching changes this)
- train stability stats (grad norm, loss spikes, expert tokens and router collapses)
- throughput (tokens/sec) + peak memory

## Architecture and Modeling

- Primary model changes to experiment with are the following:
  - patching
  - attention variants
  - QK normalization
- These will run on 10% training budget.
- Then follow up with pairwise interactions for the modeling.
- Scale up to 4096 after we figure out which variable works best
- Throw in some FP8 training
- Batch size schedule

## Reporting

- metric vs GPU-hours curve (one plot)
- metric at fixed compute points (0.25×, 0.5×, 1×)
  - Compare early changes at 10% compute budget
- compute-to-target table (time/GPU-hours to reach baseline quality)
- throughput + memory + stability notes
