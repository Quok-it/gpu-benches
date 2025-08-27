# Try multiple methods to find CUDA installation
TEMP_NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME := $(shell if [ -x "$(TEMP_NVCC)" ]; then echo $(TEMP_NVCC) | rev | cut -d'/' -f3- | rev; else echo /usr/local/cuda; fi)
NVCC := $(TEMP_NVCC)

# internal flags
NVCCFLAGS   := -std=c++11 -O3 -arch=sm_90 --compiler-options="-O2 -pipe -march=native -Wall -fopenmp" -Xcompiler -rdynamic --generate-line-info
CCFLAGS     := 
LDFLAGS     := -L$(CUDA_HOME)/lib64 -lcublas -lnvidia-ml
NAME 		:= latency
PREFIX		:=
N 			:= 1


$(PREFIX)cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)hip-$(NAME): main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	echo $(HIP_HOME)
	hipcc -std=c++17 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl -o $@ $<

clean:
	rm -f cuda-$(NAME) hip-$(NAME)
	rm -f *-cuda-$(NAME) *-hip-$(NAME)
	rm main.hip