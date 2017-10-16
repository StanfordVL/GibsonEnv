#ifndef RENDER_CUDA_F_HPP
#define RENDER_CUDA_F_HPP


extern "C" {
    uint* copyToGPU(uint * cube_idx_to_equi, size_t count);
    void projectCubeMapToEquirectangular(float * dst, float * src, uint * cube_idx_to_equi, size_t count, size_t src_size);
    float* allocateBufferOnGPU(size_t count);
    void fillBlue(float * dst, cudaArray_t src, size_t offset, size_t w, size_t h);
}

#endif
