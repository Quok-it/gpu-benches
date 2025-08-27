# Optional host compiler override. Example: make CCBIN=g++-12
CCBIN ?=
CCBIN_FLAG = $(if $(CCBIN),-ccbin $(CCBIN),)

# Try multiple methods to find CUDA installation
TEMP_NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME := $(shell if [ -x "$(TEMP_NVCC)" ]; then echo $(TEMP_NVCC) | rev | cut -d'/' -f3- | rev; else echo /usr/local/cuda; fi)
NVCC := $(TEMP_NVCC)

TEMP_HIPCC := $(shell which hipcc)
HIP_HOME :=  /opt/rocm

#(shell echo $(TEMP_HIPCC) | rev |  cut -d'/' -f4- | rev)

# internal flags
NVCCFLAGS   := -std=c++14 -O3 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_90,code=compute_90 \
               --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/lib64/\" $(CCBIN_FLAG)
CCFLAGS     := 
LDFLAGS     := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/extras/CUPTI/lib64 -lcupti -lcuda   -lnvidia-ml -lnvperf_host -lnvperf_target -lm -lstdc++
NAME 		:= cache
PREFIX		:=
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include -I$(CUDA_HOME)/include



$(PREFIX)cuda-$(NAME): main.cu Makefile
	echo $(CUDA_HOME)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)hip-$(NAME): main.hip Makefile ../rocm-metrics/rocm-metrics.hpp
	echo $(HIP_HOME)
	hipcc -std=c++17 -I$(HIP_HOME)/include/rocprofiler/ -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl -o $@ $<

clean:
	rm -f cuda-$(NAME) hip-$(NAME)