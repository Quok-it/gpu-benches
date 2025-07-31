# GPU Memory Latency Benchmark

A GPU memory latency characterization benchmark that measures access latency across different cache levels and working set sizes using pointer-chasing methodology. The pointer-chasing methodology forces sequential access and prevents prefetching so it measures pure latency. This benchmark helps understand GPU memory hierarchy latency characteristics and identify latency blockers at different cache levels.

This benchmark measures **GPU memory access latency** by running pointer-chasing kernels with varying working set sizes and measuring the latency in GPU clock cycles at each level:

- L1 cache latency (small working sets ~1-32 kB)
- L2 cache latency (medium working sets ~32-512 kB) 
- DRAM latency (large working sets >1 MB)
- Cache miss penalties and latency characteristics
- Memory hierarchy transition points

## Pointer-Chasing Kernel Operation

The benchmark uses a pointer-chasing kernel that creates dependent memory accesses:

```cuda
// Simplified version of actual core operation
for (int64_t n = 0; n < N; n += unroll_factor) {
    for (int u = 0; u < unroll_factor; u++) {
        idx = (int64_t *)*idx;  // dependent memory access - each load waits for previous
    }
}
```

| Component | Purpose |
|-----------|---------|
| **Working Set (LEN)** | Linked list size ranging from 16 elements to 16M elements |
| **Iterations** | Number of pointer chases per measurement |
| **Pointer Chain** | Randomized linked list structure forcing sequential dependent accesses |
| **Cache Line Structure** | Multiple pointers per cache line to stress cache line loading (uses all 16 pointers loaded) |
| **Unified Memory** | Uses cudaMallocManaged for simplified memory management |

## Memory Access Pattern

- **Randomized Chain**: Creates a shuffled linked list where each element points to the next in random order
- **Cache Line Optimization**: Multiple pointers per cache line (cl_size = 16 pointers per 128-byte cache line)
- **Sequential Dependencies**: Each memory access depends on the result of the previous access
- **Single Thread**: Uses single thread to ensure purely sequential access pattern
- **Working Set**: LEN × cache_line_size × sizeof(int64_t) bytes per measurement

## How to Build and Run

```bash
# Compile the benchmark
make

# Run the benchmark
./cuda-latency
```

## Output

```
     iters  freq     data    exec time   latency_avg latency_med latency_05p latency_95p
    100000  1980       2.0      12.5      246.8      246.8      246.8      246.8
    100000  1980       4.0      12.6      249.5      249.5      249.5      249.5
     16703  1980      32.0      12.7      751.2      751.2      751.2      751.2
     17444  1980      64.0      13.1      748.4      748.4      748.4      748.4
     18205  1980     128.0      14.2      774.8      774.8      774.8      774.8
```

**Column Descriptions:**
- **iters**: Number of pointer chasing iterations performed
- **freq**: GPU clock frequency in MHz
- **data**: Working set size in kB
- **exec time**: Kernel execution time in milliseconds
- **latency_avg**: Average memory access latency in GPU clock cycles
- **latency_med**: Median memory access latency in GPU clock cycles  
- **latency_05p**: 5th percentile latency in GPU clock cycles (fastest accesses)
- **latency_95p**: 95th percentile latency in GPU clock cycles (slowest accesses)

## Visualization

The `plot.py` script generates latency plots showing memory hierarchy characteristics:

```bash
python3 plot.py
```