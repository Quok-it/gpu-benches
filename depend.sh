#!/bin/bash
# dependencies for microbenchmarks

echo "Installing dependencies for microbenchmarks..."

# update package lists
apt-get update

# install build essentials
apt-get install -y build-essential

# CUDA toolkit should be pre-installed in container
# CUPTI should also be available in CUDA installation so lowk need nothing else

echo "Dependencies installation completed!"