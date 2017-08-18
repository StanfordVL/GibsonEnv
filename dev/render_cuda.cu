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


__global__ void set_float(float *depth)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
      depth[(y+j)*width + x] = 1e5;
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


__global__ void render_final(float *points3d_polar, float * depth_render, unsigned char * img, unsigned char * render)
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
     //printf("%d %d\n", x, y);
     if (points3d_polar[(ih * w + iw) * 3 + 0] < depth_render[(ty * w + tx)]) {
     render[(ty * w + tx) * 3 + 0] = img[(ih * w + iw) * 3 +0];
     render[(ty * w + tx) * 3 + 1] = img[(ih * w + iw) * 3 +1];
     render[(ty * w + tx) * 3 + 2] = img[(ih * w + iw) * 3 +2];
     depth_render[(ty * w + tx)] = points3d_polar[(ih * w + iw) * 3 + 0];
     
     }
  }
}




extern "C"{
    
void render(int h,int w,unsigned char * img, float * depth,float * pose, unsigned char * render, float * depth_render){
    //int ih, iw, i, ic;
    
    const int nx = w;
    const int ny = h;
    const int depth_mem_size = nx*ny*sizeof(float);
    const int frame_mem_size = nx*ny*sizeof(unsigned char) * 3;
    
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    unsigned char *d_img, *d_render;
    float *d_depth, *d_pose, *d_depth_render;
    float *d_3dpoint, *d_3dpoint_after, *d_3dpoint_polar;
    
    cudaMalloc((void **)&d_img, frame_mem_size);
    cudaMalloc((void **)&d_render, frame_mem_size);
    cudaMalloc((void **)&d_depth, depth_mem_size);
    cudaMalloc((void **)&d_depth_render, depth_mem_size);
    cudaMalloc((void **)&d_3dpoint, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_after, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_polar, depth_mem_size * 4);
    cudaMalloc((void **)&d_pose, sizeof(float) * 16);
    
    
    cudaMemcpy(d_pose, pose, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img, img, frame_mem_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth, depth, depth_mem_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_render, 0, frame_mem_size);
    cudaMemset(d_3dpoint, 0, depth_mem_size * 4);
    cudaMemset(d_3dpoint_after, 0, depth_mem_size * 4);
    
    to3d_point<<< dimGrid, dimBlock >>>(d_depth, d_3dpoint);
    transform<<< dimGrid, dimBlock >>>(d_3dpoint_after, d_3dpoint, d_pose);
    transform2d<<<dimGrid, dimBlock>>>(d_3dpoint_after, d_3dpoint_polar);
    
    set_float  <<< dimGrid, dimBlock >>> (d_depth_render);
    //copy_mem <<< dimGrid, dimBlock >>> (d_img, d_render);
    render_final <<< dimGrid, dimBlock >>> (d_3dpoint_polar, d_depth_render, d_img, d_render);
    
    cudaMemcpy(render, d_render, frame_mem_size, cudaMemcpyDeviceToHost);
    
    //float * point_polar = (float *)malloc(sizeof(float) * h * w * 4);
    //cudaMemcpy(point_polar, d_3dpoint_polar, depth_mem_size * 4, cudaMemcpyDeviceToHost);
    
    //int i;
    //for (i=0; i < 100; i++) printf("%f\n", point_polar[i]);
    
    cudaFree(d_img);
    cudaFree(d_depth);
    cudaFree(d_render);
    cudaFree(d_depth_render);
    cudaFree(d_3dpoint);
    cudaFree(d_3dpoint_after);
    cudaFree(d_3dpoint_polar);
    cudaFree(d_pose);

}
    
    
}//extern "C"