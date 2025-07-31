# GPU Small Kernels Benchmark

A GPU kernel launch overhead characterization benchmark that measures the performance trade-offs between kernel launch latency and compute efficiency for small workloads. This benchmark helps to understand when kernel launch overhead dominates execution time and optimal strategies for cache-friendly small kernels.

This benchmark measures **GPU small kernel performance** by running lightweight SCALE operations on varying data sizes that fit within GPU cache hierarchy:

- Kernel launch overhead vs computation time analysis
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
    
    A[tidx] = B[tidx] * 0.25;  // super simple scale operation
}
```

| Component | Purpose |
|-----------|---------|
| **Working Set Size** | Data sizes from 4KB to several MB (cache-friendly to DRAM-bound) |
| **Repeated Execution** | 10k kernel launches to measure average overhead |
| **Simple Computation** | Minimal arithmetic to isolate launch overhead effects |
| **Memory Pattern** | Streaming access pattern optimized for coalescing (multiple threads in a warp access consecutive memory addresses simultaneously) |

## Memory Arrays

- **Array A**: Output array for SCALE results
- **Array B**: Input array for SCALE operation  
- **Working Set**: 2×size elements per kernel (input + output arrays added together)
- **Memory Footprint**: Varies from around 4KB to several MB

## How to Build and Run

```bash
# Compile the benchmark
make

# Run with regular kernel launches
./cuda-small-kernels
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
- **Column 1**: Array size in elements/data items (4096, 4341, etc.)
- **Column 2**: Total memory footprint (both arrays combined)
- **Columns 3-8**: Effective bandwidth (GB/s) for different block sizes:
  - **Column 3**: 32 threads per block
  - **Column 4**: 64 threads per block  
  - **Column 5**: 128 threads per block
  - **Column 6**: 256 threads per block
  - **Column 7**: 512 threads per block
  - **Column 8**: 1024 threads per block

## Visualization

The `plot.py` script generates performance plots showing the overhead vs bandwidth trade-offs:

```bash
python3 plot.py
```