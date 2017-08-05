#include <cstdio>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

extern "C"{
    
void render(int h,int w,unsigned char * img, float * depth,float * pose, unsigned char * render){
    int ih, iw, i, ic;
    
    float * points3d = (float*) malloc(sizeof(float) * h * w * 3);
    float * points3d_after = (float*) malloc(sizeof(float) * h * w * 3);
    float * points3d_polar = (float*) malloc(sizeof(float) * h * w * 3);
    
    float * depth_render = (float *) malloc(sizeof(float) * h * w);
    
    for (i = 0; i < 5; i++) {
        printf("%f ", pose[i]);
    }
    printf("\n");
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            for (ic = 0; ic < 3; ic ++) {
                render[(ih * w + iw) * 3 +ic] = 0;
            }
            depth_render[ih * w + iw] = 1e10;
        }
    }
    
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            float depth_point = depth[ih * w + iw] * 128.0;
            //printf("%f %f\n", depth[ih * w + iw], depth_point);
            float phi = ((float)(ih) + 0.5) / float(h) * M_PI;
            float theta = ((float)(iw) + 0.5) / float(w) * 2 * M_PI;
            points3d[(ih * w + iw) * 3 + 0] = depth_point * sin(phi) * cos(theta);
            points3d[(ih * w + iw) * 3 + 1] = depth_point * sin(phi) * sin(theta);
            points3d[(ih * w + iw) * 3 + 2] = depth_point * cos(phi);
        }
    }
    
    float alpha, beta, gamma, x, y, z;
    x = pose[0];
    y = pose[1];
    z = pose[2];
    alpha = pose[4];
    beta = pose[3];
    gamma = 0;

    float transformation_matrix[9];
    
    transformation_matrix[0] = cos(alpha) * cos(beta);
    transformation_matrix[1] = cos(alpha) * sin(beta) * sin(gamma) - sin(alpha) * cos(gamma);
    transformation_matrix[2] = cos(alpha) * sin(beta) * cos(gamma) + sin(alpha) * sin(gamma);
    transformation_matrix[3] = sin(alpha) * cos(beta);
    transformation_matrix[4] = sin(alpha) * sin(beta) * sin(gamma) + cos(alpha) * cos(gamma);
    transformation_matrix[5] = sin(alpha) * sin(beta) * cos(gamma) - cos(alpha) * sin(gamma);
    transformation_matrix[6] = -sin(beta);
    transformation_matrix[7] = cos(beta) * sin(gamma);
    transformation_matrix[8] = cos(beta) * cos(gamma);
  
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            for (ic = 0; ic < 3; ic ++) {
                points3d_after[(ih * w + iw) * 3 + ic] = (points3d[(ih * w + iw) * 3 + 0]+x) * transformation_matrix[3 * ic + 0] + (points3d[(ih * w + iw) * 3 + 1]+y) * transformation_matrix[3 * ic + 1] + (points3d[(ih * w + iw) * 3 + 2]+z) * transformation_matrix[3 * ic + 2];
            }
        }
    }
    
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            float x = points3d_after[(ih * w + iw) * 3 + 0];
            float y = points3d_after[(ih * w + iw) * 3 + 1];
            float z = points3d_after[(ih * w + iw) * 3 + 2];
            
            points3d_polar[(ih * w + iw) * 3 + 0] = sqrt(x * x + y * y + z * z);
            points3d_polar[(ih * w + iw) * 3 + 1] = atan2(y, x);
            points3d_polar[(ih * w + iw) * 3 + 2] = atan2(sqrt(x * x + y * y), z);
                
        }
    }
    
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            int x = round((points3d_polar[(ih * w + iw) * 3 + 1] + M_PI)/(2*M_PI) * w - 0.5);
            int y = round((points3d_polar[(ih * w + iw) * 3 + 2])/M_PI * h - 0.5);
            //printf("%d %d\n", x, y);
            if (points3d_polar[(ih * w + iw) * 3 + 0] < depth_render[(y * w + x)]) {
                for (ic = 0; ic < 3; ic ++) {
                    render[(y * w + x) * 3 + ic] = img[(ih * w + iw) * 3 +ic];
                }
                depth_render[(y * w + x)] = points3d_polar[(ih * w + iw) * 3 + 0];
            }
        }
    }
        
    
    free(points3d);
    free(points3d_after);
    free(points3d_polar);
    free(depth_render);
}
    
    
}//extern "C"