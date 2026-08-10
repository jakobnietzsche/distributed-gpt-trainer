# Correctness

## Invariants

### 1. Identical Initial Parameters
All ranks should starts from identical parameters.

### 2. Separate Samples
Each global step should consume separate samples across ranks.

### 3. Gradient Equivalence
A synchronized gradient should equal the arithmetic mean of the local gradients,
unless the configured loss scaling specifies otherwise.

### 4. Identical Optimizer Updates
All ranks should apply the same optimizer update in the same order.

### 5. Valid Checkpoints
Only a fully written checkpoint (i.e., checkpoint can't fail partway through) may be treated
as resumable.