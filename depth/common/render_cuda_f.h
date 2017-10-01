#ifndef RENDER_CUDA_F_HPP
#define RENDER_CUDA_F_HPP


extern "C" {
    uint* move_idxs_to_gpu(uint * cube_idx_to_equi, size_t count);
    void cube_to_equi(float * dst, float * src, uint * cube_idx_to_equi, size_t count, size_t src_size);
    float* allocate_buffer_on_gpu(size_t count);
    void fillBlue(float * dst, cudaArray_t src, size_t offset);
}

#endif
