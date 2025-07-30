# GPU L2 Cache Specific Benchmark

A GPU L2 cache characterization benchmark that measures cache performance under varying levels of inter-block contention (where multiple thread blocks compete for the same shared resource) and spatial locality. This benchmark specifically targets L2 cache behaviour by keeping per-block working sets constant while varying the total memory footprint across multiple thread blocks.

This benchmark measures **GPU L2 cache performance** by testing how effectively the L2 cache handles multiple concurrent memory streams from different thread blocks:

- L2 cache capacity and bandwidth under multi-block access
- Cache contention effects as working set size increases
- Spatial locality impact on L2 cache performance
- Transition point where L2 cache becomes saturated
- DRAM bandwidth when L2 cache is overwhelmed

## L2 Cache Kernel Operation

The benchmark uses a specialized kernel that creates controlled spatial access patterns:

```cuda
// Simplified version of main operation
for (int i = 0; i < N / 2; i++) {
    int idx = (blockDim.x * blockRun * i + (blockIdx.x % blockRun) * BLOCKSIZE) * 2 + threadIdx.x;
    localSum += B[idx] * B[idx + BLOCKSIZE];  // spatially separated reads
}
```

| Component | Purpose |
|-----------|---------|
| **Fixed Working Set (N=64)** | 512 kB per thread block (constant) |
| **Variable blockRun** | Controls spatial separation and total footprint |
| **Spatial Pattern** | Accesses B[idx] and B[idx + BLOCKSIZE] for locality testing |
| **Multiple Blocks** | 200k blocks create heavy L2 cache pressure |

## Memory Access Pattern

- **Per-Block Data**: Fixed 512 kB working set per thread block
- **Total Working Set**: 512 kB × blockRun (varies from ~1.5 MB to ~5 GB)
- **Access Pattern**: Spatially separated reads within each block's region
- **L2 Contention**: Multiple blocks compete for L2 cache space

## How to Build and Run (Nvidia)

```bash
# Compile the benchmark
make

# Run the benchmark (requires sudo for performance counters)
sudo ./cuda-l2-cache
```

## Output

```
     data set   exec time     spread       Eff. bw
       512 kB   3032064 kB        34ms       1.5%    3118.5 GB/s     3119 GB/s      0 GB/s   3119 GB/s      0 GB/s 
       512 kB   3335168 kB        34ms       1.5%    3099.4 GB/s     3099 GB/s      0 GB/s   3099 GB/s      0 GB/s 
       512 kB   3668480 kB        34ms       1.6%    3085.5 GB/s     3086 GB/s      0 GB/s   3086 GB/s      0 GB/s 
       512 kB   4035072 kB        34ms       1.6%    3065.5 GB/s     3065 GB/s      0 GB/s   3065 GB/s      0 GB/s 
       512 kB   4438528 kB        34ms       0.8%    3053.2 GB/s     3053 GB/s      1 GB/s   3053 GB/s      0 GB/s 
       512 kB   4881920 kB        35ms       0.0%    3030.6 GB/s     3031 GB/s      1 GB/s   3031 GB/s      0 GB/s 
```

**Column Descriptions:**
- **data set (1st)**: Fixed per-block working set (always 512 kB)
- **data set (2nd)**: Total memory footprint across all blocks  
- **exec time**: Kernel execution time in milliseconds
- **spread**: Measurement variance as percentage
- **Eff. bw**: Effective bandwidth in GB/s
- **DRAM read/write**: DRAM traffic 
- **L2 read/store**: L2 cache traffic

## Visualization

The `plot.py` script generates L2 cache performance plots:

```bash
python3 plot.py
```