# Try multiple methods to find CUDA installation
TEMP_NVCC := $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CUDA_HOME := $(shell if [ -x "$(TEMP_NVCC)" ]; then echo $(TEMP_NVCC) | rev | cut -d'/' -f3- | rev; else echo /usr/local/cuda; fi)
NVCC := $(TEMP_NVCC)

# internal flags
NVCCFLAGS   := -std=c++14 -O3 -arch=sm_90 --compiler-options="-O2 -pipe -Wall -fopenmp -g" -Xcompiler -rdynamic --generate-line-info  -Xcompiler \"-Wl,-rpath,$(CUDA_HOME)/extras/CUPTI/lib64/\" -Xcompiler "-Wall"
CCFLAGS     := 
NAME 		:= stream
LDFLAGS     := -L$(CUDA_HOME)/lib64 -lcuda -lnvidia-ml
PREFIX		:=
INCLUDES 	:=  -I$(CUDA_HOME)/extras/CUPTI/include

$(PREFIX)cuda-$(NAME): main.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

$(PREFIX)$(NAME)-gsl: main_gsl.cu Makefile
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)


main.hip: main.cu
	hipify-perl main.cu > main.hip

$(PREFIX)hip-$(NAME): main.hip Makefile
	hipcc -std=c++17  -I$(HIP_HOME)/hsa/include/hsa -L$(HIP_HOME)/rocprofiler/lib -lhsa-runtime64 -ldl -o $@ $<

clean:
	rm -f cuda-$(NAME) hip-$(NAME) *-hip-$(NAME) *-cuda-$(NAME) main.hip