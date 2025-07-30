# GPU L2 Stream Benchmark

A GPU memory streaming benchmark that specifically targets **L2 cache behaviour** across different memory access lengths. This benchmark measures sustained GPU memory bandwidth while testing how the L2 cache responds to varying streaming patterns and buffer sizes.

This benchmark measures **GPU L2 cache streaming performance** by systematically varying the memory access length to find optimal cache utilization patterns. It tests how efficiently the GPU can stream data through different cache levels under varying memory footprint conditions:

- Different memory access lengths 
- Different thread configurations and occupancy levels  
- L2 cache hit/miss behavior across footprint sizes
- Sustained streaming performance vs buffer size

## Memory Kernels

The benchmark includes 4 streaming kernels that test different memory access patterns:

| Kernel | Memory Streams | Operation | Purpose |
|--------|----------------|-----------|---------|
| **read** | 1 (read) | `temp = B[i % length]` | Pure read bandwidth through L2 |
| **scale** | 2 (1R+1W) | `A[i % length] = B[i % length] * 1.2` | Read-modify-write with L2 |
| **triad** | 4 (3R+1W) | `A[i % length] = B[i % length] * D[i % length] + C[i % length]` | Max memory pressure |
| **write** | 1 (write) | `A[i % length] = 0.23` | Pure write bandwidth through L2 |

## Key Features

- **Variable Length Access**: Tests various memory footprints 
- **L2 Cache Targeting**: Access patterns designed to stress L2 cache behavior
- **Modulo Access**: Uses `i % length` indexing to create repeating access patterns
- **Occupancy Control**: Uses shared memory "spoiler" to control SM occupancy
- **Measurements**: Reports min/median/max bandwidth for each kernel

## Memory Arrays

- **A, B, C, D**: Four GPU arrays, each around 128MB (512MB total GPU memory)
- **Access Length**: Variable length parameter controls effective working set size
- **Access Pattern**: Modulo indexing creates repeating streaming patterns within the length

## How to Build and Run

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-l2-stream
```

## Output Format

```
clock: 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 1980 
  96    1    12672      277      5%     |  GB/s:    444    444    444    608    608    608   1082   1082   1082    436    435    435 
 112    1    14784      277    5.5%     |  GB/s:    514    514    514    703    702    702   1250   1250   1248    506    506    505 
  64    2    16896      277    6.2%     |  GB/s:    584    584    584    823    823    823   1449   1449   1448    571    571    571 
```

### Column Explanation:
- **Column 1**: Thread block size
- **Column 2**: Blocks per SM  
- **Column 3**: Total threads across all SMs
- **Column 4**: Access length in elements (controls working set size)
- **Column 5**: SM occupancy percentage
- **GB/s values**: 12 bandwidth measurements (3 per kernel):
  - **444 444 444**: read kernel (min/median/max GB/s)
  - **608 608 608**: scale kernel (min/median/max GB/s)  
  - **1082 1082 1082**: triad kernel (min/median/max GB/s)
  - **436 435 435**: write kernel (min/median/max GB/s)

## L2 Cache Analysis

This benchmark is particularly useful for:

- **Cache Working Set Analysis**: Finding optimal buffer sizes for L2 cache
- **Streaming Pattern Optimization**: Understanding how access length affects bandwidth
- **Memory Hierarchy Performance**: Comparing L2 vs DRAM bandwidth across footprints
- **Application Tuning**: Determining optimal tile sizes for cache-friendly algorithms

## Visualization

The `plot.py` script generates performance plots showing bandwidth vs access length:

```bash
python3 plot.py
```

The plots help visualize:
- L2 cache working set transitions
- Bandwidth cliffs when exceeding L2 capacity  
- Optimal streaming lengths for different kernels
- Performance scaling across thread configurations