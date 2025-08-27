# runs gpu-stream, gpu-cache, gpu-l2-cache, cuda-memcpy, cuda-matmul, cuda-incore, gpu-small-kernels (~1.5 hours)

#!/bin/bash
set -e

# load env variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# get GPU UUID
GPU_UUID=$(nvidia-smi -q | grep -i "GPU UUID" | awk '{print $NF}')
if [ -z "$GPU_UUID" ]; then
    echo "Error: Could not get GPU UUID"
    exit 1
fi

# get execution ID from env var or latest from db
if [ -z "$EXECUTION_ID" ]; then
    echo "EXECUTION_ID not set, fetching latest from db..."
    EXECUTION_ID=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "
    SELECT execution_id 
    FROM benchmark_executions 
    ORDER BY started_at DESC 
    LIMIT 1;" | xargs)
    
    if [ -z "$EXECUTION_ID" ]; then
        echo "Error: No executions found in database"
        exit 1
    fi
    
    echo "Using latest execution ID from db: $EXECUTION_ID"
else
    echo "Using execution ID from environment: $EXECUTION_ID"
fi

echo "Using GPU UUID: $GPU_UUID"

# set default directory if not provided
BENCHMARK_DIR=${BENCHMARK_DIR:-$(pwd)}

cd "$BENCHMARK_DIR"

# using SQL export
echo "Starting GPU benchmarks collection..."
echo "Results will be inserted directly into database"

echo "================================================"
echo "======== Memory Microbenchmarks ================"
echo "================================================"

# gpu-stream benchmark
echo ""
echo "=== GPU Stream Microbenchmark ==="
cd memory/gpu-stream
make clean && make
echo "Running GPU Stream benchmark and inserting results into database..."
./cuda-stream "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

echo "GPU Stream microbenchmark completed and added to database!"
cd ../..

# gpu-cache benchmark (needs sudo)
echo ""
echo "=== GPU Cache Microbenchmark ==="
cd memory/gpu-cache
make clean && make
if sudo -n true 2>/dev/null; then
    echo "Running GPU Cache benchmark and inserting results into database..."
    sudo ./cuda-cache "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    
    echo "GPU Cache microbenchmark completed and added to database!"
else
    echo "GPU Cache microbenchmark skipped (requires sudo)"
fi
cd ../..

# gpu-l2-cache benchmark (needs sudo)
echo ""
echo "=== GPU L2 Cache Microbenchmark ==="
cd memory/gpu-l2-cache
make clean && make
if sudo -n true 2>/dev/null; then
    echo "Running GPU L2 Cache benchmark and inserting results into database..."
    sudo ./cuda-l2-cache "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"
    
    echo "GPU L2 Cache microbenchmark completed and added to database!"
else
    echo "GPU L2 Cache microbenchmark skipped (requires sudo)"
fi
cd ../..

# cuda-memcpy benchmark
echo ""
echo "=== CUDA Memory Copy Microbenchmark ==="
cd memory/cuda-memcpy
make clean && make
echo "Running CUDA Memory Copy benchmark and inserting results into database..."
./cuda-memcpy "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

echo "CUDA Memory Copy microbenchmark completed and added to database!"
cd ../..

echo "================================================"
echo "======== Compute Microbenchmarks ==============="
echo "================================================"

# cuda-matmul benchmark
echo ""
echo "=== CUDA Matrix Multiplication Microbenchmark ==="
cd compute/cuda-matmul
make clean && make
echo "Running CUDA Matrix Multiplication benchmark and inserting results into database..."
./cuda-matmul "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

echo "CUDA Matrix Multiplication microbenchmark completed and added to database!"
cd ../..

# cuda-incore benchmark
echo ""
echo "=== CUDA In-Core Compute Microbenchmark ==="
cd compute/cuda-incore
make clean && make
echo "Running CUDA In-Core benchmark and inserting results into database..."
./cuda-incore "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

echo "CUDA In-Core microbenchmark completed and added to database!"
cd ../..

echo "================================================"
echo "======== System Microbenchmarks ================"
echo "================================================"

# gpu-small-kernels benchmark
echo ""
echo "=== GPU Small Kernels System Microbenchmark ==="
cd system/gpu-small-kernels
make clean && make
echo "Running GPU Small Kernels benchmark and inserting results into database..."
./cuda-small-kernels "$EXECUTION_ID" "$GPU_UUID" | PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME"

echo "GPU Small Kernels microbenchmark completed and added to database!"
cd ../..

echo ""
echo "All GPU benchmarks completed successfully!"
echo "Results have been inserted directly into database."
