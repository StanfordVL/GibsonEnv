nvcc -Xcompiler -fPIC -shared -o render_cuda.so render_cuda.cu
nvcc -Xcompiler -fPIC -shared -o occinf.so occinf.cu
