#!/bin/bash
set -e

cd "$BENCHMARK_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR/gpu-benches"

echo "Starting GPU benchmarks collection..."
echo "Results will be saved to: $RESULTS_DIR/gpu-benches"

# GPU Cache benchmark (needs sudo for performance counters)
echo ""
echo "=== GPU Cache Benchmark ==="
cd gpu-cache
make clean && make
if sudo -n true 2>/dev/null; then
    echo "Running with performance counters (sudo available)"
    sudo ./cuda-cache > "$RESULTS_DIR/gpu-benches/gpu-cache-results.txt"
else
    echo "Running without performance counters (no sudo)"
    ./cuda-cache > "$RESULTS_DIR/gpu-benches/gpu-cache-results.txt"
fi
echo "GPU Cache benchmark completed"
cd ..

# GPU L2 Cache benchmark  
echo ""
echo "=== GPU L2 Cache Benchmark ==="
cd gpu-l2-cache
make clean && make
if sudo -n true 2>/dev/null; then
    echo "Running with performance counters (sudo available)"
    sudo ./cuda-l2-cache > "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results.txt"
else
    echo "Running without performance counters (no sudo)"
    ./cuda-l2-cache > "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results.txt"
fi
echo "GPU L2 Cache benchmark completed"
cd ..

# CUDA Memory Copy benchmark
echo ""
echo "=== CUDA Memory Copy Benchmark ==="
cd cuda-memcpy
make clean && make
./cuda-memcpy > "$RESULTS_DIR/gpu-benches/cuda-memcpy-results.txt"
echo "CUDA Memory Copy benchmark completed"
cd ..

# Unified Memory Stream benchmark
echo ""
echo "=== Unified Memory Stream Benchmark ==="
cd um-stream
make clean && make
./um-stream > "$RESULTS_DIR/gpu-benches/um-stream-results.txt"
echo "Unified Memory Stream benchmark completed"
cd ..

# Copy README files for reference
echo ""
echo "Copying documentation files..."
find . -name "README.md" -exec cp {} "$RESULTS_DIR/gpu-benches/" \; 2>/dev/null || true

echo ""
echo "All GPU benchmarks completed successfully!"
echo "Results saved to: $RESULTS_DIR/gpu-benches/"