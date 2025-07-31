# GPU Small Kernels Benchmark

A GPU kernel launch overhead characterization benchmark that measures the performance trade-offs between kernel launch latency and compute efficiency for small workloads. This benchmark helps understand when kernel launch overhead dominates execution time and optimal strategies for cache-friendly small kernels.

This benchmark measures **GPU small kernel performance** by running lightweight SCALE operations on varying data sizes that fit within GPU cache hierarchy:

- Kernel launch overhead vs. computation time analysis
- Cache-friendly workload performance (L1/L2 cache-sized datasets)
- Thread block size optimization for small kernels  
- Multiple execution strategies: regular launches, CUDA Graphs, persistent threads
- Bandwidth efficiency across different working set sizes

## Small Kernel Operation

The benchmark uses a simple memory streaming operation repeated many times:

```cuda
// Core SCALE operation
__global__ void scale(T *A, const T *B, const int size) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx >= size) return;
    
    A[tidx] = B[tidx] * 0.25;  // Simple scale operation
}
```

| Component | Purpose |
|-----------|---------|
| **Working Set Size** | Data sizes from 4KB to several MB (cache-friendly to DRAM-bound) |
| **Repeated Execution** | 10,000 kernel launches to measure average overhead |
| **Simple Computation** | Minimal arithmetic to isolate launch overhead effects |
| **Memory Pattern** | Streaming access pattern optimized for coalescing |

## Execution Modes

The benchmark supports multiple execution strategies:

| Mode | Command | Purpose |
|------|---------|---------|
| **Regular Launches** | `./cuda-small-kernels` | Standard kernel launch overhead |
| **CUDA Graphs** | `./cuda-small-kernels -graph` | Reduced launch overhead via pre-recorded graphs |
| **Persistent Threads (Atomic)** | `./cuda-small-kernels -pta` | Eliminate launch overhead with atomic synchronization |
| **Persistent Threads (Grid Sync)** | `./cuda-small-kernels -pt-gsync` | Eliminate launch overhead with grid-level synchronization |

## Memory Arrays

- **Array A**: Output array for SCALE results
- **Array B**: Input array for SCALE operation  
- **Working Set**: 2×size elements per kernel (input + output arrays)
- **Memory Footprint**: Varies from ~4KB to several MB

## How to Build and Run

```bash
# Compile the benchmark
make

# Run with regular kernel launches
./cuda-small-kernels

# Run with CUDA Graphs (reduced overhead)  
./cuda-small-kernels -graph

# Run with persistent threads (minimal overhead)
./cuda-small-kernels -pt-gsync
```

## Output Format

```
4096  64kB     17     17     17     17     17     17  
4341  67kB     18     18     18     18     18     18  
4601  71kB     19     19     19     19     19     19  
4877  76kB     20     20     20     20     20     20  
5169  80kB     21     21     21     21     21     21  
```

### Column Explanation:
- **Column 1**: Array size in elements (4096, 4341, etc.)
- **Column 2**: Total memory footprint (both arrays combined)
- **Columns 3-8**: Effective bandwidth (GB/s) for different block sizes:
  - **Column 3**: 32 threads per block
  - **Column 4**: 64 threads per block  
  - **Column 5**: 128 threads per block
  - **Column 6**: 256 threads per block
  - **Column 7**: 512 threads per block
  - **Column 8**: 1024 threads per block

## Performance Analysis

### Understanding Results:

1. **Small Working Sets** (< 64KB): Performance limited by kernel launch overhead
2. **Medium Working Sets** (64KB - 512KB): L2 cache performance dominates  
3. **Large Working Sets** (> 1MB): DRAM bandwidth becomes the bottleneck
4. **Block Size Effects**: Larger blocks typically better for small kernels (reduce launch overhead per thread)

### Typical Patterns:

- **Flat bandwidth across block sizes**: Launch overhead dominated  
- **Increasing bandwidth with block size**: Better amortization of fixed costs
- **Performance plateaus**: Cache or DRAM bandwidth limits reached
- **CUDA Graphs improvement**: Reduced launch overhead for repeated patterns

## Performance Model

The benchmark fits results to the analytical model:

**T = V/(a + V/b)**

Where:
- **T**: Total execution time
- **V**: Data volume (working set size)  
- **a**: Launch overhead parameter (seconds)
- **b**: Peak bandwidth parameter (GB/s)

This model separates launch overhead effects from bandwidth limitations.

## Use Cases

This benchmark is valuable for:

- **Small Kernel Optimization**: Understanding when launch overhead matters
- **Execution Strategy Selection**: Choosing between regular launches, graphs, or persistent threads
- **Cache Blocking Analysis**: Finding optimal tile sizes for cache-friendly algorithms  
- **Thread Configuration**: Optimizing block sizes for launch overhead vs. occupancy
- **GPU Architecture Comparison**: Measuring launch overhead differences across GPUs

## Visualization

The `plot.py` script generates performance plots showing the overhead vs. bandwidth trade-offs:

```bash
python3 plot.py
```

Plots typically show:
- Bandwidth vs. working set size curves
- Launch overhead inflection points
- Optimal working set sizes for different execution modes