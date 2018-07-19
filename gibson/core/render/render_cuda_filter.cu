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

//const bool pano = false;

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

__global__ void selection_sum_weights(float * selection_sum,  float * selection, int n, int stride) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int idx = 0;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
        selection_sum[((y+j)*width + x)] = 0;
        for ( idx = 0; idx < n; idx ++) {
            atomicAdd(&(selection_sum[((y+j)*width + x)]),  selection[idx * stride + ((y+j)*width + x)]);
        }
    }
}



__global__ void merge(unsigned char * img_all, unsigned char * img, float * selection, int n, int stride)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int idx = 0;
    float sum = 0;
    float weight = 0;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
      sum = 0;
      for (idx = 0; idx < n; idx ++) sum += selection[idx * stride + ((y+j)*width + x)];

      for (idx = 0; idx < n; idx ++) selection[idx * stride + ((y+j)*width + x)] /= (sum + 1e-5);

      img[3*((y+j)*width + x)] = 0;        
      img[3*((y+j)*width + x)+1] = 0;    
      img[3*((y+j)*width + x)+2] = 0;    
      
      for (idx = 0; idx < n; idx ++) {

      //weight = selection[idx * stride + ((y+j)*width + x)];
      weight = 0.25;
      //weight = 0.5;


      img[3*((y+j)*width + x)] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x)] * weight);
      img[3*((y+j)*width + x)+1] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x) + 1] * weight);
      img[3*((y+j)*width + x)+2] += (unsigned char)(img_all[idx * stride * 3 + 3*((y+j)*width + x) + 2] * weight);

      }


    }
}


__global__ void merge_sum(unsigned char * img_all, unsigned char * img, float * selection, float * selection_sum,  int n, int stride)
{
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int idx = 0;
    float weight = 0;
    for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {

      img[3*((y+j)*width + x)] = 0;
      img[3*((y+j)*width + x)+1] = 0;
      img[3*((y+j)*width + x)+2] = 0;

      for (idx = 0; idx < n; idx ++) {

      weight = selection[idx * stride + ((y+j)*width + x)] / selection_sum[((y+j)*width + x)];
      //weight = 0.25;
      //weight = 0.5;


      img[3*((y+j)*width + x)] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x)] * weight);
      img[3*((y+j)*width + x)+1] += (unsigned char) (img_all[idx * stride * 3 + 3*((y+j)*width + x) + 1] * weight);
      img[3*((y+j)*width + x)+2] += (unsigned char)(img_all[idx * stride * 3 + 3*((y+j)*width + x) + 2] * weight);

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

//#define FOV_SCALE 1.73205080757
//#define FOV_SCALE 1


__global__ void transform2d(float *points3d_after, float fov_scale)
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

      points3d_after[(ih * w + iw) * 3 + 0] = x;//sqrt(x * x + y * y + z * z);
    //points3d_after[(ih * w + iw) * 3 + 1] = atan2(y, x);
    //points3d_after[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);

      float x2 = fov_scale * x;
      if ((x2 > 0) && (y < x2 * 1.1) && (y > -x2 * 1.1) && (z < x2 * 1.1) && (z > -x2 * 1.1)) {
          points3d_after[(ih * w + iw) * 3 + 1] = y / (x2 + 1e-5);
          points3d_after[(ih * w + iw) * 3 + 2] = -z / (x2 + 1e-5);
      }
      else {
          points3d_after[(ih * w + iw) * 3 + 1] = -1;
          points3d_after[(ih * w + iw) * 3 + 2] = -1;
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

__global__ void get_average(unsigned char * img, int * nz, int * average, int scale)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  //int h = width /2;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     
     if (img[3*(ih*width + iw)] + img[3*(ih*width + iw)+1] + img[3*(ih*width + iw)+2] > 0)
     {
         //nz[ih/3 * width + iw/3] += 1;
         //average[3*(ih/3*width + iw/3)] += (int)img[3*(ih*width + iw)];
         //average[3*(ih/3*width + iw/3)+1] += (int)img[3*(ih*width + iw)+1];
         //average[3*(ih/3*width + iw/3)+2] += (int)img[3*(ih*width + iw)+2];
         
         atomicAdd(&(nz[ih/scale * width + iw/scale]), 1);
         atomicAdd(&(average[3*(ih/scale*width + iw/scale)]), (int)img[3*(ih*width + iw)]);
         atomicAdd(&(average[3*(ih/scale*width + iw/scale)+1]), (int)img[3*(ih*width + iw)+1]);
         atomicAdd(&(average[3*(ih/scale*width + iw/scale)+2]), (int)img[3*(ih*width + iw)+2]);
         
     }
     
  }
}


__global__ void fill_with_average(unsigned char *img, int * nz, int * average, int scale)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;
  //int h = width /2;
  
  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
     int iw = x;
     int ih = y + j;
     
     if ((img[3*(ih*width + iw)] + img[3*(ih*width + iw)+1] + img[3*(ih*width + iw)+2] == 0) && (nz[ih/scale * width + iw/scale] > 0))
     {
         img[3*(ih*width + iw)] = (unsigned char)(average[3*(ih/scale*width + iw/scale)] / nz[ih/scale * width + iw/scale]);
         img[3*(ih*width + iw) + 1] = (unsigned char)(average[3*(ih/scale*width + iw/scale) + 1] / nz[ih/scale * width + iw/scale]);
         img[3*(ih*width + iw) + 2] = (unsigned char)(average[3*(ih/scale*width + iw/scale) + 2] / nz[ih/scale * width + iw/scale]);
     }
     
  }
}




__global__ void render_final(float *points3d_polar, float * selection, float * depth_render, int * img,  int * render, int oh, int ow)
{
 int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int w = gridDim.x * TILE_DIM;
  int h = w /2;
  int maxsize = oh * ow;

  for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
  {
  
     int iw = x;
     int ih = y + j;

          
     int tx = round((points3d_polar[(ih * w + iw) * 3 + 1] + 1)/2 * ow - 0.5);
     int ty = round((points3d_polar[(ih * w + iw) * 3 + 2] + 1)/2 * oh - 0.5);
          
     float tx_offset = ((points3d_polar[(ih * w + iw) * 3 + 1] + 1)/2 * ow - 0.5);
     float ty_offset = ((points3d_polar[(ih * w + iw) * 3 + 2] + 1)/2 * oh - 0.5);
     
     float tx00 = 0;
     float ty00 = 0;
     
     float tx01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
     float ty01 = ((points3d_polar[(ih * w + iw + 1) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;
     
     float tx10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
     float ty10 = ((points3d_polar[((ih + 1) * w + iw) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;
     
     float tx11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 1] + 1)/2 * ow - 0.5) - tx_offset;
     float ty11 = ((points3d_polar[((ih+1) * w + iw + 1) * 3 + 2] + 1)/2 * oh - 0.5) - ty_offset;
     
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
     int delta00 = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]) - (int)(100 * depth_render[(ty * ow + tx)]);
     int delta01 = (int)(12800/128 * points3d_polar[(ih * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[(ty * ow + tx + 1)]);
     int delta10 = (int)(12800/128 * points3d_polar[((ih + 1) * w + iw) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * ow + tx)]);
     int delta11 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw + 1) * 3 + 0]) - (int)(100 * depth_render[((ty+1) * ow + tx + 1)]);
     
     int mindelta = min(min(delta00, delta01), min(delta10, delta11));
     int maxdelta = max(max(delta00, delta01), max(delta10, delta11));

     int depth00 = (int)(12800/128 * points3d_polar[(ih * w + iw) * 3 + 0]);
     int depth01 = (int)(12800/128 * points3d_polar[(ih * w + iw + 1) * 3 + 0]);
     int depth10 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw) * 3 + 0]);
     int depth11 = (int)(12800/128 * points3d_polar[((ih+1) * w + iw+1) * 3 + 0]);
     int max_depth =  max(max(depth00, depth10), max(depth01, depth11));
     int min_depth =  min(min(depth00, depth10), min(depth01, depth11));
     int delta_depth = max_depth - min_depth;

     int txmin = floor(tx_offset + min(min(tx00, tx11), min(tx01, tx10)));
     int txmax = ceil(tx_offset + max(max(tx00, tx11), max(tx01, tx10)));
     int tymin = floor(ty_offset + min(min(ty00, ty11), min(ty01, ty10)));
     int tymax = ceil(ty_offset + max(max(ty00, ty11), max(ty01, ty10)));
      
     float newx, newy;
     int r,g,b;
     int itx, ity;
     
     //render[(ty * ow + tx)] = img[ih * w + iw];
     //selection[(ty * ow + tx)] = 1.0;

     float tolerance = 0.1 * this_depth > 10? 0.1 * this_depth : 10;
     float tolerance2 = 0.05 * max_depth > 10? 0.05 * max_depth: 10;

     float flank = 0.01;
     if ((delta_depth < tolerance2) && (y > 1 * h/8) && (y < (h*7)/8))
     if (((mindelta > - tolerance) && (maxdelta <  tolerance)) && (this_depth < 10000)) {
           if (((txmax - txmin) * (tymax - tymin) < 1600) && (txmax - txmin < 40) && (tymax - tymin < 40))
           {
               for (itx = txmin; itx < txmax; itx ++)
                   for (ity = tymin; ity < tymax; ity ++)
                   { if (( 0 <= itx) && (itx < ow) && ( 0 <= ity) && (ity < oh))
                       {
                       newx = (itx - tx_offset) * it00 + it10 * (ity - ty_offset);
                       newy = (itx - tx_offset) * it01 + it11 * (ity - ty_offset);

                       //printf("%f %f\n", newx, newy);
                       if ((newx > -flank) && (newx < 1 + flank) && (newy > -flank) && (newy < 1 + flank))
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

                           if ((ity * ow + itx > 0) && (ity * ow + itx < maxsize)) {
                               render[(ity * ow + itx)] = r * 256 * 256 + g * 256 + b;
                               selection[(ity * ow + itx)] = 1.0 / abs(det);
                           }
                           }
                       }
                   }
                       
            }
     }

  }
  
  
  
}


extern "C"{
    
void render(int n, int h,int w, int oh, int ow, unsigned char * img, float * depth,float * pose, unsigned char * render, float * depth_render, float fov){
    //int ih, iw, i, ic;
    //printf("inside cuda code %d\n", depth);
    
    //printf("scale %d\n", s);
    const int nx = w;
    const int ny = h;
    
    const int onx = ow;
    const int ony = oh;
    
    const size_t input_mem_size = nx*ny;
    const size_t output_mem_size = onx * ony;
    
    dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
    dim3 dimGrid_out(onx/TILE_DIM, ony/TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

    unsigned char *d_img, *d_render, *d_render_all;
    float *d_depth, *d_pose;
    float *d_depth_render;
    float *d_3dpoint, *d_3dpoint_after;
    
    float * d_selection, * d_selection_sum;
    int * nz;
    int * average;
    
    int *d_render2, *d_img2;
    
    cudaMalloc((void **)&d_img, input_mem_size * sizeof(unsigned char) * 3);
    cudaMalloc((void **)&d_img2, input_mem_size * sizeof(int));
    cudaMalloc((void **)&d_render, output_mem_size * sizeof(unsigned char) * 3);
    cudaMalloc((void **)&d_render_all, output_mem_size * sizeof(unsigned char) * 3 * n);
    cudaMalloc((void **)&d_depth, input_mem_size * sizeof(float));
    cudaMalloc((void **)&d_depth_render, output_mem_size * sizeof(float));
    cudaMalloc((void **)&d_3dpoint, input_mem_size * sizeof(float) * 4);
    cudaMalloc((void **)&d_3dpoint_after, input_mem_size * sizeof(float) * 4);
    cudaMalloc((void **)&d_pose, sizeof(float) * 16);


    cudaMalloc((void **)&d_selection, output_mem_size * sizeof(float) * n);
    cudaMalloc((void **)&d_selection_sum, output_mem_size * sizeof(float));

    cudaMalloc((void **)&d_render2, output_mem_size * sizeof(int));
    
    
    cudaMalloc((void **)&nz, output_mem_size * sizeof(int));
    cudaMalloc((void **)&average, output_mem_size * sizeof(int) * 3);
    

    cudaMemset(nz, 0, output_mem_size * sizeof(int));
    cudaMemset(average, 0, output_mem_size * sizeof(int) * 3);
    cudaMemset(d_selection, 0, output_mem_size * sizeof(float) * n);
    cudaMemset(d_selection_sum, 0, output_mem_size * sizeof(float));

    cudaMemcpy(d_depth_render, depth_render, output_mem_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_render_all, 0, output_mem_size * sizeof(unsigned char) * 3 * n);

    int idx;
    for (idx = 0; idx < n; idx ++) {
    
        cudaMemcpy(d_pose, &(pose[idx * 16]), sizeof(float) * 16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_img, &(img[idx * input_mem_size * 3]), input_mem_size * sizeof(unsigned char) * 3, cudaMemcpyHostToDevice);
        cudaMemcpy(d_depth, &(depth[idx * input_mem_size]), input_mem_size * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(d_render, 0, output_mem_size * sizeof(unsigned char) * 3);
        cudaMemset(d_render2, 0, output_mem_size * sizeof(int));
        
        cudaMemset(d_img2, 0, input_mem_size * sizeof(int));  
        cudaMemset(d_3dpoint, 0, input_mem_size * sizeof(float) * 4);
        cudaMemset(d_3dpoint_after, 0, input_mem_size * sizeof(float) * 3);

        to3d_point<<< dimGrid, dimBlock >>>(d_depth, d_3dpoint);
        transform<<< dimGrid, dimBlock >>>(d_3dpoint_after, d_3dpoint, d_pose);

        float fov_scale = tan(fov/2);

        transform2d<<<dimGrid, dimBlock>>>(d_3dpoint_after, fov_scale);

        char_to_int <<< dimGrid, dimBlock >>> (d_img2, d_img);

        render_final <<< dimGrid, dimBlock >>> (d_3dpoint_after, &(d_selection[idx * onx * ony]),  d_depth_render, d_img2, d_render2, oh, ow);
        
        //int_to_char <<< dimGrid_out, dimBlock >>> (d_render2, d_render);
        int_to_char <<< dimGrid_out, dimBlock >>> (d_render2, &(d_render_all[idx * output_mem_size * 3]));


        int fill_size[1] = {3};

        for (int j = 0; j < 1; j++) {
            cudaMemset(nz, 0, output_mem_size * sizeof(int));
            cudaMemset(average, 0, output_mem_size * sizeof(int) * 3);
            get_average <<< dimGrid_out, dimBlock >>> (&(d_render_all[idx * output_mem_size * 3]), nz, average, fill_size[j]);
            fill_with_average <<< dimGrid_out, dimBlock >>> (&(d_render_all[idx * output_mem_size * 3]), nz, average, fill_size[j]);
        }
    }

        selection_sum_weights <<< dimGrid_out, dimBlock >>> (d_selection_sum, d_selection, n, output_mem_size);
        merge_sum <<< dimGrid_out, dimBlock >>> (d_render_all, d_render, d_selection, d_selection_sum, n, output_mem_size);
        //merge <<< dimGrid_out, dimBlock >>> (d_render_all, d_render, d_selection, n, output_mem_size);

    cudaMemcpy(render, d_render, output_mem_size * sizeof(unsigned char) * 3 , cudaMemcpyDeviceToHost);

        
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
    cudaFree(d_selection);
    cudaFree(nz);
    cudaFree(average);
    cudaFree(d_selection_sum);
}
    
    
}//extern "C"
