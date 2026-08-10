# Protocol for Benchmarks

## 1. Warmup
Every benchmark should undergo the same number of warmup steps (the default is 10).
This is to exclude initial iterations that may be unrepresentative of steady-state performance.
Warmup helps exclude one-time overheads such as memory allocation, caching, and initialization
of libraries or GPU kernels.

## 2. Metrics and Units
All benchmarks should report metrics using consistent units so results can be compared across runs.

- Training throughput: tokens / second
- Step latency: milliseconds / step
- Training loss: unitless; reported every 1 million tokens
- Memory usage: megabytes (MB)

## 3. Repetitions
Separate from the warmup steps, each benchmark will have 5 repetitions with 100 measured steps per repetition.

One measured step consists of a forward pass, loss computation, backpropagation, any required synchronization,
optimizer update, and metric collection.

## 4. Synchronization Points
Each repetition has two synchronization points: one immediately before measurement begins and one immediately
after the final measured step. These ensure all relevant workers and devices have completed outstanding work
before timing begins or ends.