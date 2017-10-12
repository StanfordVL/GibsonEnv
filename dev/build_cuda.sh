nvcc -Xcompiler -fPIC -shared -o render_cuda.so render_cuda.cu
nvcc -Xcompiler -fPIC -shared -o render_cuda_f.so render_cuda_filter.cu
nvcc -Xcompiler -fPIC -shared -o occinf.so occinf.cu
