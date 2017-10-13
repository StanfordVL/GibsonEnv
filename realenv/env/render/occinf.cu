#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

using namespace std;

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void copy_mem(unsigned char *source, unsigned char *render)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      for (int channel = 0; channel < 3; channel ++ )
        render[3*((y+j)*width + x) + channel] = source[3 * ((y+j)*width + x) + channel];
}


__global__ void set_depth(unsigned int *depth)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      depth[(y+j)*width + x] = 65535;
}


__global__ void char_to_int(int * img2, unsigned char * img)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      img2[(y+j)*width + x] =  img[3*((y+j)*width + x) + 0] * 256 * 256 + img[3*((y+j)*width + x) + 1] * 256 + img[3*((y+j)*width + x) + 2];
}


__global__ void int_to_char(int * img2, unsigned char * img)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
      img[3*((y+j)*width + x)] = img2[(y+j)*width + x] / (256*256);
      img[3*((y+j)*width + x)+1] = img2[(y+j)*width + x] / 256 % 256;
      img[3*((y+j)*width + x)+2] = img2[(y+j)*width + x] % 256;
    }
}


__global__ void to3d_point(float *depth, float *points3d)
{
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  int h = w / 2;
    
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     
     int iw = x;
     int ih = y + j;
     float depth_point = depth[ih*w + iw] * 128.0;
     float phi = ((float)(ih) + 0.5) / float(h) * M_PI;
     float theta = ((float)(iw) + 0.5) / float(w) * 2 * M_PI + M_PI;
  
      points3d[(ih * w + iw) * 4 + 0] = depth_point * sin(phi) * cos(theta);
      points3d[(ih * w + iw) * 4 + 1] = depth_point * sin(phi) * sin(theta);
      points3d[(ih * w + iw) * 4 + 2] = depth_point * cos(phi);
      points3d[(ih * w + iw) * 4 + 3] = 1;
  
  }
}

__global__ void transform(float *points3d_after, float *points3d, float * transformation_matrix)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     for (int ic = 0; ic < 3; ic ++) {
     points3d_after[(ih * w + iw) * 3 + ic] = points3d[(ih * w + iw) * 4 + 0] * transformation_matrix[4 * ic + 0]
     + points3d[(ih * w + iw) * 4 + 1] * transformation_matrix[4 * ic + 1] 
     + points3d[(ih * w + iw) * 4 + 2] * transformation_matrix[4 * ic + 2] 
     + points3d[(ih * w + iw) * 4 + 3] * transformation_matrix[4 * ic + 3]; 
    }
  }
}


__global__ void transform2d(float *points3d_after, float *points3d_polar)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     float x = points3d_after[(ih * w + iw) * 3 + 0];
     float y = points3d_after[(ih * w + iw) * 3 + 1];
     float z = points3d_after[(ih * w + iw) * 3 + 2];

    points3d_polar[(ih * w + iw) * 3 + 0] = sqrt(x * x + y * y + z * z);
    points3d_polar[(ih * w + iw) * 3 + 1] = atan2(y, x);
    points3d_polar[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);
  }
}


__global__ void render_depth(float *points3d_polar, unsigned int * depth_render)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  int h = w /2;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int ty = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h - 0.5);
     int this_depth = (int)(100 * points3d_polar[(ih * w + iw) * 3 + 0]);
     atomicMin(&depth_render[(ty * w + tx)] , this_depth);
  }
}




__global__ void render_occu(float *points3d_polar, unsigned int * depth_render, bool * occu_map, float * depth)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  int h = w /2;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int ty = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h - 0.5);
     
     float depth_point = depth[ih*w + iw] * 100.0 * 128;
     
     int this_depth = (int)(100 * points3d_polar[(ih * w + iw) * 3 + 0]);
     if ((this_depth - depth_render[(ty * w + tx)]) > 5) {
     
         int scale = int(depth_point / float(this_depth) * 4);
         
         for (int j = -scale; j < scale; j ++)
             for (int k = -scale; k < scale; k++)
                 {
                 if ((ty + j < h) && (ty + j > 0) && (tx + k > 0) && (tx + k < w))
                     occu_map[((ty+j) * w + (tx+k))] = 1;
                 };
     }
  }
}




extern "C"{
    
void occinf(int h,int w, float * depth,float * pose, bool * occmap, unsigned int * depth_render){
    const int nx = w;
    const int ny = h;
    const int depth_mem_size = nx*ny*sizeof(float);
    const int occ_map_size = nx*ny*sizeof(bool);
    
    
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    float *d_depth, *d_pose;
    unsigned int *d_depth_render;
    float *d_3dpoint, *d_3dpoint_after, *d_3dpoint_polar;
    bool *d_occu;
    
    cudaMalloc((void **)&d_depth, depth_mem_size);
    cudaMalloc((void **)&d_depth_render, nx * ny * sizeof(unsigned int));
    cudaMalloc((void **)&d_3dpoint, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_after, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_polar, depth_mem_size * 4);
    cudaMalloc((void **)&d_pose, sizeof(float) * 16);
    cudaMalloc((void **)&d_occu, occ_map_size);

    
    cudaMemcpy(d_depth_render, depth_render, nx * ny * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pose, pose, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth, depth, depth_mem_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_3dpoint, 0, depth_mem_size * 4);
    cudaMemset(d_3dpoint_after, 0, depth_mem_size * 4);
    cudaMemset(d_occu, 0, occ_map_size);
    
    
    to3d_point<<< dimGrid, dimBlock >>>(d_depth, d_3dpoint);
    transform<<< dimGrid, dimBlock >>>(d_3dpoint_after, d_3dpoint, d_pose);
    transform2d<<<dimGrid, dimBlock>>>(d_3dpoint_after, d_3dpoint_polar);
        
    render_depth <<< dimGrid, dimBlock >>> (d_3dpoint_polar, d_depth_render);
    render_occu <<< dimGrid, dimBlock >>> (d_3dpoint_polar, d_depth_render, d_occu, d_depth);
    cudaMemcpy(occmap, d_occu, occ_map_size, cudaMemcpyDeviceToHost);
    
    
    cudaFree(d_occu);
    cudaFree(d_depth);
    cudaFree(d_depth_render);
    cudaFree(d_3dpoint);
    cudaFree(d_3dpoint_after);
    cudaFree(d_3dpoint_polar);
    cudaFree(d_pose);
}
    
    
}//extern "C"