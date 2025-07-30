# GPU STREAM Benchmark

A GPU adaptation of the STREAM memory bandwidth benchmark - this is designed to measure sustained GPU memory bandwidth across different computational patterns and thread configs.

This benchmark measures **GPU memory bandwidth** by streaming data through GPU memory using various computational kernels. It tests how efficiently the GPU can move data between memory and compute units under different conditions:

- Different thread counts (1K to 400K+ threads)
- Different occupancy levels (0.8% to 100% SM utilization)
- Different memory access patterns
- Sustained vs peak memory performance

## Memory Kernels

The benchmark includes 6 kernels based on the original STREAM operations:

| Kernel | Memory Streams | Operation | Purpose |
|--------|----------------|-----------|---------|
| **init** | 1 (write) | `A[i] = 0.23` | Pure write bandwidth |
| **read** | 1 (read) | `temp = B[i]` | Pure read bandwidth |
| **scale** | 2 (1R+1W) | `A[i] = B[i] * 1.2` | Read-modify-write |
| **triad** | 4 (3R+1W) | `A[i] = B[i] * D[i] + C[i]` | Maximum memory pressure |
| **3pt stencil** | 2 (neighbors) | `A[i] = 0.5*B[i-1] - B[i] + 0.5*B[i+1]` | Spatial locality |
| **5pt stencil** | 2 (wide neighbors) | Similar but ±2 neighbors | Complex spatial access |

## Memory Arrays

- **A, B, C, D**: Four GPU arrays, each  around 256MB (2GB total GPU memory)
- **A**: Always output array
- **B**: Primary input array  
- **C, D**: Additional inputs for multi-operand operations

## How to Build and Run (Nvidia)

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-stream

## Output

```
block smBlocks   threads    occ%   |                init       read       scale     triad       3pt        5pt
  32      3456       1    1.6%     |  GB/s:         106         42         80        142         76         74
  64      6912       1    3.1%     |  GB/s:         210         85        159        280        146        143
```

- **block**: Thread block size
- **smBlocks**: Total threads across all SMs
- **threads**: Blocks per SM
- **occ%**: SM occupancy percentage
- **GB/s columns**: Memory bandwidth for each kernel

## Occupancy Control

The benchmark uses a "spoiler" shared memory mechanism to artificially control occupancy, allowing measurement across different thread configurations to find optimal performance points.

## Visualization

The `plot.py` script generates performance plots:

```bash
python3 plot.py
```