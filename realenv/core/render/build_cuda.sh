nvcc -Wno-deprecated-gpu-targets -Xcompiler -fPIC -shared -o render_cuda.so render_cuda.cu
nvcc -Wno-deprecated-gpu-targets -Xcompiler -fPIC -shared -o render_cuda_f.so render_cuda_filter.cu
nvcc -Wno-deprecated-gpu-targets -Xcompiler -fPIC -shared -o occinf.so occinf.cu
