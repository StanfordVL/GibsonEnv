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

#ifndef max

#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )

#endif



#ifndef min

#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )

#endif


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


__global__ void merge(unsigned char * img_all, unsigned char * img, int n, int stride)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int idx = 0;
    
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
      
      int nz = 0;
      for (idx = 0; idx < n; idx ++) 
          if (img_all[stride * idx + 3*((y+j)*width + x)] + img_all[stride * idx + 3*((y+j)*width + x) + 1] + img_all[stride * idx + 3*((y+j)*width + x) + 2] > 0)
          nz +=1 ;
          
      img[3*((y+j)*width + x)] = 0;        
      img[3*((y+j)*width + x)+1] = 0;    
      img[3*((y+j)*width + x)+2] = 0;    
      
      
      if (nz > 0)
      for (idx = 0; idx < n; idx ++) {
      
      img[3*((y+j)*width + x)] += img_all[idx * stride + 3*((y+j)*width + x)] / nz;
      img[3*((y+j)*width + x)+1] += img_all[idx * stride + 3*((y+j)*width + x) + 1] / nz;
      img[3*((y+j)*width + x)+2] += img_all[idx * stride + 3*((y+j)*width + x) + 2] / nz;
      
      }
      
      
    
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
     float depth_point = depth[ ih*w + iw ] * 128.0;
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


__global__ void transform2d(float *points3d_after)
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

    points3d_after[(ih * w + iw) * 3 + 0] = sqrt(x * x + y * y + z * z);
    points3d_after[(ih * w + iw) * 3 + 1] = atan2(y, x);
    points3d_after[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);
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
     int this_depth = (int)(512 * points3d_polar[(ih * w + iw) * 3 + 0]);
     atomicMin(&depth_render[(ty * w + tx)] , this_depth);
  }
}



__global__ void render_final(float *points3d_polar, int * depth_render, int * img,  int * render)
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
          
     int txlu = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int tylu = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h - 0.5);
     
     int txld = round((points3d_polar[(ih * w + iw + 1) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int tyld = round((points3d_polar[(ih * w + iw + 1) * 3 + 2])/M_PI * h - 0.5);
     
     int txru = round((points3d_polar[((ih + 1) * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int tyru = round((points3d_polar[((ih + 1) * w + iw) * 3 + 2])/M_PI * h - 0.5);
     
     int txrd = round((points3d_polar[((ih+1) * w + iw + 1) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
     int tyrd = round((points3d_polar[((ih+1) * w + iw + 1) * 3 + 2])/M_PI * h - 0.5);
     
     
     int this_depth = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]);
     int delta = this_depth - depth_render[(ty * w + tx)];
     
     int txmin = min(min(txlu, txrd), min(txru, txld));
     int txmax = max(max(txlu, txrd), max(txru, txld));
     int tymin = min(min(tylu, tyrd), min(tyru, tyld));
     int tymax = max(max(tylu, tyrd), max(tyru, tyld));
      
     if ((y > h/8) && (y < (h*7)/8))
     if ((delta > -15) && (delta < 15) && (this_depth < 10000)) {
           
           if ((txmax - txmin) * (tymax - tymin) < 50)
           {
               for (tx = txmin; tx < txmax; tx ++)
                   for (ty = tymin; ty < tymax; ty ++)
                       render[(ty * w + tx)] = img[(ih * w + iw)];
            }
     }
  }
}


extern "C"{
    
void render(int n, int h,int w,unsigned char * img, float * depth,float * pose, unsigned char * render, int * depth_render){
    //int ih, iw, i, ic;
    printf("inside cuda code %d\n", depth);
    const int nx = w;
    const int ny = h;
    const size_t depth_mem_size = nx*ny*sizeof(float);
    const size_t frame_mem_size = nx*ny*sizeof(unsigned char) * 3;
    
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    unsigned char *d_img, *d_render, *d_render_all;
    float *d_depth, *d_pose;
    int *d_depth_render;
    float *d_3dpoint, *d_3dpoint_after;
    
    int *d_render2, *d_img2;
    
    cudaMalloc((void **)&d_img, frame_mem_size);
    cudaMalloc((void **)&d_render, frame_mem_size);
    cudaMalloc((void **)&d_render_all, frame_mem_size * n);
    cudaMalloc((void **)&d_depth, depth_mem_size);
    cudaMalloc((void **)&d_depth_render, nx * ny * sizeof(int));
    cudaMalloc((void **)&d_3dpoint, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_after, depth_mem_size * 4);
    cudaMalloc((void **)&d_pose, sizeof(float) * 16);
    cudaMalloc((void **)&d_render2, nx * ny * sizeof(int));
    cudaMalloc((void **)&d_img2, nx * ny * sizeof(int));
    cudaMemcpy(d_depth_render, depth_render, nx * ny * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemset(d_render_all, 0, frame_mem_size * n);
    
    int idx;
    for (idx = 0; idx < n; idx ++) {
    
        cudaMemcpy(d_pose, &(pose[idx * 16]), sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img, &(img[idx * nx * ny * 3]), frame_mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_depth, &(depth[idx * nx * ny]), depth_mem_size, cudaMemcpyHostToDevice);

        //int i;
        //for (i = 0; i < 100; i++) {
        //    printf("%f ", depth[i + idx * nx * ny]);
        //}
        //printf("\n");

        cudaMemset(d_render, 0, frame_mem_size);
        cudaMemset(d_render2, 0, nx * ny * sizeof(int));
        cudaMemset(d_img2, 0, nx * ny * sizeof(int));  
        cudaMemset(d_3dpoint, 0, depth_mem_size * 4);
        cudaMemset(d_3dpoint_after, 0, depth_mem_size * 4);

        to3d_point<<< dimGrid, dimBlock >>>(d_depth, d_3dpoint);
        transform<<< dimGrid, dimBlock >>>(d_3dpoint_after, d_3dpoint, d_pose);
        transform2d<<<dimGrid, dimBlock>>>(d_3dpoint_after);

        char_to_int <<< dimGrid, dimBlock >>> (d_img2, d_img);

        render_final <<< dimGrid, dimBlock >>> (d_3dpoint_after, d_depth_render, d_img2, d_render2);
        int_to_char <<< dimGrid, dimBlock >>> (d_render2, d_render);
        int_to_char <<< dimGrid, dimBlock >>> (d_render2, &(d_render_all[idx * nx * ny * 3]));

    }

        merge <<< dimGrid, dimBlock >>> (d_render_all, d_render, n, nx * ny * 3);
        cudaMemcpy(render, d_render, frame_mem_size, cudaMemcpyDeviceToHost);
        
        
    cudaFree(d_img);
    cudaFree(d_depth);
    cudaFree(d_render2);
    cudaFree(d_img2);
    cudaFree(d_render);
    cudaFree(d_depth_render);
    cudaFree(d_3dpoint);
    cudaFree(d_3dpoint_after);
    cudaFree(d_pose);
    cudaFree(d_render_all);
}
    
    
}//extern "C"