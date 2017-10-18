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

const bool pano = false;

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



__global__ void fill(unsigned char * img)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
      
      if ( img[3*((y+j)*width + x)] + img[3*((y+j)*width + x)+1] + img[3*((y+j)*width + x)+2] == 0) {
          
          img[3*((y+j)*width + x)] = img[3*((y+j)*width + x + 1)];
          img[3*((y+j)*width + x)+1] = img[3*((y+j)*width + x + 1)+1];
          img[3*((y+j)*width + x)+2] = img[3*((y+j)*width + x + 1)+2];
      }  
      
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
    //points3d_after[(ih * w + iw) * 3 + 1] = atan2(y, x);
    //points3d_after[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);
      if ((x > 0) && (y < x) && (y > -x) && (z < x) && (z > -x)) {
          points3d_after[(ih * w + iw) * 3 + 1] = y / (x + 1e-5);
          points3d_after[(ih * w + iw) * 3 + 2] = -z / (x + 1e-5);
      }
      else {
          points3d_after[(ih * w + iw) * 3 + 1] = 0;
          points3d_after[(ih * w + iw) * 3 + 2] = 0;
      }
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



__global__ void render_final(float *points3d_polar, float * depth_render, int * img,  int * render, int s)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  int h = w /2;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     //int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w * s - 0.5);
     //int ty = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h * s - 0.5);
          
     int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + 1)/2 * h * s - 0.5);
     int ty = round((points3d_polar[(ih * w + iw) * 3 + 2] + 1)/2 * h * s - 0.5);
          
     float tx_offset = ((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w * s - 0.5);
     float ty_offset = ((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h * s - 0.5);
     
     float tx00 = 0;
     float ty00 = 0;
     
     float tx01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 1] + M_PI)/(2*M_PI) * w * s - 0.5) - tx_offset;
     float ty01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 2])/M_PI * h * s - 0.5) - ty_offset;
     
     float tx10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w * s - 0.5) - tx_offset;
     float ty10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 2])/M_PI * h * s - 0.5) - ty_offset;
     
     float tx11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 1] + M_PI)/(2*M_PI) * w * s - 0.5) - tx_offset;
     float ty11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 2])/M_PI * h * s - 0.5) - ty_offset;
     
     float t00 = 0 * (float)tx00 + (float)tx01 * -1.0/3  + (float)tx10 *  2.0/3   + (float)tx11 *  1.0/3;
     float t01 = 0 * (float)ty00 + (float)ty01 * -1.0/3  + (float)ty10 *  2.0/3   + (float)ty11 *  1.0/3;
     float t10 = 0 * (float)tx00 + (float)tx01 *  2.0/3  + (float)tx10 * -1.0/3   + (float)tx11 *  1.0/3;
     float t11 = 0 * (float)ty00 + (float)ty01 *  2.0/3  + (float)ty10 * -1.0/3   + (float)ty11 *  1.0/3;
     
     float det = t00 * t11 - t01 * t10 + 1e-10;
     
     //printf("%f %f %f %f %f\n", t00, t01, t10, t11, det);
     
     float it00, it01, it10, it11;
     
     it00 = t11/det;
     it01 = -t01/det;
     it10 = -t10/det;
     it11 = t00/det;
     
     //printf("inverse %f %f %f %f\n", it00, it01, it10, it11);
     
     int this_depth = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]);
     int delta00 = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]) - (int)(100 * depth_render[(ty * w * s + tx)]);
     int delta01 = (int)(12800/128 * points3d_polar[(ih * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[(ty * w * s + tx + 1)]);
     int delta10 = (int)(12800/128 * points3d_polar[((ih + 1) * w + iw) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * w * s + tx)]);
     int delta11 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * w * s + tx + 1)]);
     
     int mindelta = min(min(delta00, delta01), min(delta10, delta11));
     int maxdelta = max(max(delta00, delta01), max(delta10, delta11));
     
     
     int txmin = floor(tx_offset + min(min(tx00, tx11), min(tx01, tx10)));
     int txmax = ceil(tx_offset + max(max(tx00, tx11), max(tx01, tx10)));
     int tymin = floor(ty_offset + min(min(ty00, ty11), min(ty01, ty10)));
     int tymax = ceil(ty_offset + max(max(ty00, ty11), max(ty01, ty10)));
      
     float newx, newy;
     int r,g,b;
     int itx, ity;
     
     render[(ty * w * s + tx)] = img[ih * w + iw];
     
     /*
     
     if ((y > h/8) && (y < (h*7)/8))
     if ((mindelta > -10) && (maxdelta < 10) && (this_depth < 10000)) {
           if ((txmax - txmin) * (tymax - tymin) < 100 * s * s)
           {
               for (itx = txmin; itx < txmax; itx ++)
                   for (ity = tymin; ity < tymax; ity ++)
                       {
                       newx = (itx - tx_offset) * it00 + it10 * (ity - ty_offset);
                       newy = (itx - tx_offset) * it01 + it11 * (ity - ty_offset);
                       
                       //printf("%f %f\n", newx, newy);
                       if ((newx > -0.05) && (newx < 1.05) && (newy > -0.05) && (newy < 1.05))
                          { 
                          if (newx < 0) newx = 0;
                          if (newy < 0) newy = 0;
                          if (newx > 1) newx = 1;
                          if (newy > 1) newy = 1;
                          
                          
                           r = img[(ih * w + iw)] / (256*256) * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] / (256*256) * (1-newx) * (newy) + img[((ih+1) * w + iw)] / (256*256) * (newx) * (1-newy) + img[((ih+1) * w + iw + 1)] / (256*256) * newx * newy;
                           g = img[(ih * w + iw)] / 256 % 256 * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] / 256 % 256 * (1-newx) * (newy) + img[((ih+1) * w + iw)] / 256 % 256  * (newx) * (1-newy)  + img[((ih+1) * w + iw + 1)] / 256 % 256 * newx * newy;
                           b = img[(ih * w + iw)] % 256 * (1-newx) * (1-newy) + img[(ih * w + iw + 1)] % 256 * (1-newx) * (newy) + img[((ih+1) * w + iw)] % 256 * (newx) * (1-newy)  + img[((ih+1) * w + iw + 1)] % 256 * newx * newy ;
                           
                           if (r > 255) r = 255;
                           if (g > 255) g = 255;
                           if (b > 255) b = 255;
                           
                           render[(ity * w * s + itx)] = r * 256 * 256 + g * 256 + b;
                           }
                       }
                       
            }
     }*/
  }
  
  
  
}


extern "C"{
    
void render(int n, int h,int w, int s, unsigned char * img, float * depth,float * pose, unsigned char * render, float * depth_render){
    //int ih, iw, i, ic;
    //printf("inside cuda code %d\n", depth);
    
    printf("scale %d\n", s);
    const int nx = w/s;
    const int ny = h/s;
    const size_t depth_mem_size = nx*ny*sizeof(float);
    const size_t frame_mem_size = nx*ny*sizeof(unsigned char) * 3;
    
    const size_t render_mem_size = nx * ny * s * s;
    
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimGrid2(nx * s/TILE_DIM, ny * s/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    unsigned char *d_img, *d_render, *d_render_all;
    float *d_depth, *d_pose;
    float *d_depth_render;
    float *d_3dpoint, *d_3dpoint_after;
    
    int *d_render2, *d_img2;
    
    cudaMalloc((void **)&d_img, frame_mem_size);
    cudaMalloc((void **)&d_render, render_mem_size * sizeof(unsigned char) * 3);
    cudaMalloc((void **)&d_render_all, render_mem_size * sizeof(unsigned char) * 3 * n);
    cudaMalloc((void **)&d_depth, depth_mem_size);
    cudaMalloc((void **)&d_depth_render, render_mem_size * sizeof(float));
    cudaMalloc((void **)&d_3dpoint, depth_mem_size * 4);
    cudaMalloc((void **)&d_3dpoint_after, depth_mem_size * 4);
    cudaMalloc((void **)&d_pose, sizeof(float) * 16);
    cudaMalloc((void **)&d_render2, render_mem_size * sizeof(int));
    cudaMalloc((void **)&d_img2, render_mem_size * sizeof(int));
    
    
    cudaMemcpy(d_depth_render, depth_render, render_mem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_render_all, 0, render_mem_size * sizeof(unsigned char) * 3 * n);
    
    int idx;
    for (idx = 0; idx < n; idx ++) {
    
        cudaMemcpy(d_pose, &(pose[idx * 16]), sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img, &(img[idx * nx * ny * 3]), frame_mem_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_depth, &(depth[idx * nx * ny]), depth_mem_size, cudaMemcpyHostToDevice);

        cudaMemset(d_render, 0, render_mem_size * sizeof(unsigned char) * 3);
        cudaMemset(d_render2, 0, render_mem_size * sizeof(int));
        
        cudaMemset(d_img2, 0, nx * ny * sizeof(int));  
        cudaMemset(d_3dpoint, 0, depth_mem_size * 4);
        cudaMemset(d_3dpoint_after, 0, depth_mem_size * 4);

        to3d_point<<< dimGrid, dimBlock >>>(d_depth, d_3dpoint);
        transform<<< dimGrid, dimBlock >>>(d_3dpoint_after, d_3dpoint, d_pose);
        transform2d<<<dimGrid, dimBlock>>>(d_3dpoint_after);

        char_to_int <<< dimGrid, dimBlock >>> (d_img2, d_img);

        render_final <<< dimGrid, dimBlock >>> (d_3dpoint_after, d_depth_render, d_img2, d_render2, s);
        
        int_to_char <<< dimGrid2, dimBlock >>> (d_render2, d_render);
        int_to_char <<< dimGrid2, dimBlock >>> (d_render2, &(d_render_all[idx * nx * ny * s * s * 3]));

        fill <<< dimGrid2, dimBlock >>> (&(d_render_all[idx * nx * ny * s * s * 3]));
    }

        merge <<< dimGrid2, dimBlock >>> (d_render_all, d_render, n, nx * ny * s * s * 3);
        
        cudaMemcpy(render, d_render, render_mem_size * sizeof(unsigned char) * 3 , cudaMemcpyDeviceToHost);
        
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
