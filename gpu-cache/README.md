# GPU Cache Hierarchy Benchmark

A GPU memory hierarchy characterization benchmark that measures effective bandwidth across different cache levels and working set sizes. This benchmark helps understand GPU cache behavior and identify optimal data sizes for different cache levels.

This benchmark measures **GPU cache hierarchy performance** by running compute kernels with varying working set sizes and measuring the effective bandwidth at each level:

- L1 cache performance (small working sets ~1-32 kB)
- L2 cache performance (medium working sets ~32-512 kB) 
- DRAM performance (large working sets >1 MB)
- Cache transition points and bandwidth characteristics
- Memory access pattern efficiency

## Cache Kernel Operation

The benchmark uses a compute-intensive kernel that performs repeated memory accesses:

```cuda
// Simplified version of the actual core operation
for (int iter = 0; iter < iters; iter++) {
    for (int i = 0; i < N; i += BLOCKSIZE) {
        localSum += B[i] * B2[i];  // two memory reads & multiply-accumulate
    }
}
```

| Component | Purpose |
|-----------|---------|
| **Working Set (N)** | Data size per SM ranging from 1kB to 280MB |
| **Iterations** | Repeated accesses to stress cache hierarchy |
| **Compute Operations** | Multiply-accumulate to prevent pure memory-bound behavior (tests more realistic scenario, prevents cache bypass, measures sustained performance) |
| **Memory Pattern** | Strided access (thread access memory at regular intervals) with two arrays (B and B2) |

## Memory Arrays

- **Array A**: Output array for results (prevents dead code elimination - prevents optimization)
- **Array B**: Primary input array for cache testing
- **Array B2**: Secondary input array (B + N offset) to force more cache lines to be loaded
- **Working Set**: 2×N elements per kernel invocation (array B and array B2)

## How to Build and Run (Nvidia)

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-cache
```

## Output

```
     data set   exec time     spread        Eff. bw       DRAM read      DRAM write         L2 read       L2 store
         1 kB        71ms       0.3%   14944.8 GB/s         0 GB/s          0 GB/s          0 GB/s          0 GB/s 
        32 kB        34ms       6.6%   33263.5 GB/s         0 GB/s          0 GB/s          0 GB/s          0 GB/s 
       280 kB        76ms      75.1%   27931.0 GB/s         0 GB/s          0 GB/s          0 GB/s          0 GB/s 
     10380 kB       155ms      43.4%   11045.5 GB/s         0 GB/s          0 GB/s          0 GB/s          0 GB/s 
    280692 kB       164ms      14.0%    7714.7 GB/s         0 GB/s          0 GB/s          0 GB/s          0 GB/s 
```

**Column Descriptions:**
- **data set**: Working set size per SM in kB
- **exec time**: Kernel execution time in milliseconds
- **spread**: Measurement variance as percentage
- **Eff. bw**: Effective bandwidth in GB/s (computed from working set size and time)
- **DRAM read/write**: DRAM traffic 
- **L2 read/store**: L2 cache traffic

## Visualization

The `plot.py` script generates performance plots showing cache hierarchy behavior:

```bash
python3 plot.py
```