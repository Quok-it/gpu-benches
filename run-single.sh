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

echo "Using GPU UUID: $GPU_UUID"

# set default directories if not provided
BENCHMARK_DIR=${BENCHMARK_DIR:-$(pwd)}
RESULTS_DIR=${RESULTS_DIR:-$(pwd)/results}

cd "$BENCHMARK_DIR"

# create results directory
mkdir -p "$RESULTS_DIR/gpu-benches"

# gpu-stream now uses SQL export, other benchmarks still use txt
echo "Starting GPU benchmarks collection..."
echo "Results will be saved to: $RESULTS_DIR/gpu-benches"

# create new execution entry and get execution_id
echo "Creating new benchmark execution entry..."
EXECUTION_ID=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -q -c "
INSERT INTO benchmark_executions (execution_name, scale_type, benchmark_suite, provider, gpu_type_filter, status, started_at, total_gpus) 
VALUES ('7 microbenchmarks', 'gpu', 'microbenchmarks', 'unknown_provider', 'unknown', 'running', NOW(), 1) 
RETURNING execution_id;" | xargs)

echo "Created execution with ID: $EXECUTION_ID"

echo "================================================"
echo "======== Memory Microbenchmarks ================"
echo "================================================"

# # gpu-stream benchmark
# echo ""
# echo "=== GPU Stream Microbenchmark ==="
# cd memory/gpu-stream
# make clean && make
# ./cuda-stream "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/gpu-stream-results.sql"

# # insert into database
# echo "Inserting GPU Stream results into database..."
# PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-stream-results.sql"

# # mark execution as completed
# PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
# UPDATE benchmark_executions 
# SET status = 'completed', completed_at = NOW() 
# WHERE execution_id = $EXECUTION_ID;"

# echo "GPU Stream microbenchmark completed and added to database!"
# cd ../..

# # gpu-cache benchmark (needs sudo)
# echo ""
# echo "=== GPU Cache Microbenchmark ==="
# cd memory/gpu-cache
# make clean && make
# if sudo -n true 2>/dev/null; then
#     sudo ./cuda-cache "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/gpu-cache-results.sql"
    
#     # insert into database
#     echo "Inserting GPU Cache results into database..."
#     PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-cache-results.sql"
    
#     echo "GPU Cache microbenchmark completed and added to database!"
# else
#     echo "GPU Cache microbenchmark skipped (requires sudo)"
# fi
# cd ../..

# # gpu-l2-cache benchmark (needs sudo)
# echo ""
# echo "=== GPU L2 Cache Microbenchmark ==="
# cd memory/gpu-l2-cache
# make clean && make
# if sudo -n true 2>/dev/null; then
#     sudo ./cuda-l2-cache "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results.sql"
    
#     # insert into database
#     echo "Inserting GPU L2 Cache results into database..."
#     PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-l2-cache-results.sql"
    
#     echo "GPU L2 Cache microbenchmark completed and added to database!"
# else
#     echo "GPU L2 Cache microbenchmark skipped (requires sudo)"
# fi
# cd ../..

# # cuda-memcpy benchmark
# echo ""
# echo "=== CUDA Memory Copy Microbenchmark ==="
# cd memory/cuda-memcpy
# make clean && make
# ./cuda-memcpy "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/cuda-memcpy-results.sql"

# # insert into database
# echo "Inserting CUDA Memory Copy results into database..."
# PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-memcpy-results.sql"

# echo "CUDA Memory Copy microbenchmark completed and added to database!"
# cd ../..

echo "================================================"
echo "======== Compute Microbenchmarks ==============="
echo "================================================"

# # cuda-matmul benchmark
# echo ""
# echo "=== CUDA Matrix Multiplication Microbenchmark ==="
# cd compute/cuda-matmul
# make clean && make
# ./cuda-matmul "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/cuda-matmul-results.sql"

# # insert into database
# echo "Inserting CUDA Matrix Multiplication results into database..."
# PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-matmul-results.sql"

# echo "CUDA Matrix Multiplication microbenchmark completed and added to database!"
# cd ../..

# # cuda-incore benchmark
# echo ""
# echo "=== CUDA In-Core Compute Microbenchmark ==="
# cd compute/cuda-incore
# make clean && make
# ./cuda-incore "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/cuda-incore-results.sql"

# # insert into database
# echo "Inserting CUDA In-Core results into database..."
# PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/cuda-incore-results.sql"

# echo "CUDA In-Core microbenchmark completed and added to database!"
# cd ../..

echo "================================================"
echo "======== System Microbenchmarks ================"
echo "================================================"

# gpu-small-kernels benchmark
echo ""
echo "=== GPU Small Kernels System Microbenchmark ==="
cd system/gpu-small-kernels
make clean && make
./cuda-small-kernels "$EXECUTION_ID" "$GPU_UUID" > "$RESULTS_DIR/gpu-benches/gpu-small-kernels-results.sql"

# insert into database
echo "Inserting GPU Small Kernels results into database..."
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" < "$RESULTS_DIR/gpu-benches/gpu-small-kernels-results.sql"

echo "GPU Small Kernels microbenchmark completed and added to database!"
cd ../..

# mark execution as completed
PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "
UPDATE benchmark_executions 
SET status = 'completed', completed_at = NOW() 
WHERE execution_id = $EXECUTION_ID;"

# Copy README files for reference
echo ""
echo "Copying documentation files..."
find . -name "README.md" -exec cp {} "$RESULTS_DIR/gpu-benches/" \; 2>/dev/null || true

echo ""
echo "All GPU benchmarks completed successfully!"
echo "Results saved to: $RESULTS_DIR/gpu-benches/"