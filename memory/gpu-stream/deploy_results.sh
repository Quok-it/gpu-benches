#!/bin/bash

# Get GPU UUID
GPU_UUID=$(nvidia-smi -q | grep "GPU UUID" | cut -d' ' -f4)
if [ -z "$GPU_UUID" ]; then
    echo "Error: Could not get GPU UUID"
    exit 1
fi

echo "Using GPU UUID: $GPU_UUID"

# Compile and run benchmark
make
./cuda-stream > gpu_stream_results.sql

# Replace placeholders
sed -i "s/UPDATE_GPU_UUID/$GPU_UUID/g" gpu_stream_results.sql

# Insert into database
echo "Inserting results into database..."
psql -h benchmark-postgresql-db.cmfkgwq6gehx.us-east-1.rds.amazonaws.com -p 5432 -U postgres -d benchmarks < gpu_stream_results.sql

echo "Results deployed successfully!"