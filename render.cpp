#include <cstdio>
#include <vector>
#include <algorithm>
#include <math.h>
using namespace std;

extern "C"{
    
void render(int h,int w,unsigned char * img, float * depth,float * pose, unsigned char * render, float * depth_render){
    int ih, iw, i, ic;
    
    float * points3d = (float*) malloc(sizeof(float) * h * w * 4);
    float * points3d_after = (float*) malloc(sizeof(float) * h * w * 3);
    float * points3d_polar = (float*) malloc(sizeof(float) * h * w * 3);
        
    //for (i = 0; i < 5; i++) {
    //    printf("%f ", pose[i]);
    //}
    //printf("\n");
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
            float theta = ((float)(iw) + 0.5) / float(w) * 2 * M_PI + M_PI;
            points3d[(ih * w + iw) * 4 + 0] = depth_point * sin(phi) * cos(theta);
            points3d[(ih * w + iw) * 4 + 1] = depth_point * sin(phi) * sin(theta);
            points3d[(ih * w + iw) * 4 + 2] = depth_point * cos(phi);
            points3d[(ih * w + iw) * 4 + 3] = 1;
        }
    }
    
    float alpha, beta, gamma, x, y, z;
    x = pose[0];
    y = pose[1];
    z = pose[2];
    alpha = pose[4];
    beta = pose[3];
    gamma = pose[5];

    float transformation_matrix[16];

    for (i = 0; i < 16; i++) transformation_matrix[i] = pose[i];
  
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            for (ic = 0; ic < 3; ic ++) {
                points3d_after[(ih * w + iw) * 3 + ic] = points3d[(ih * w + iw) * 4 + 0] * transformation_matrix[4 * ic + 0]
                                                        +points3d[(ih * w + iw) * 4 + 1] * transformation_matrix[4 * ic + 1] 
                                                        +points3d[(ih * w + iw) * 4 + 2] * transformation_matrix[4 * ic + 2] 
                                                        +points3d[(ih * w + iw) * 4 + 3] * transformation_matrix[4 * ic + 3]; 
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
        
    for (ih = 0; ih < h; ih ++ ) {
        for (iw = 0; iw < w; iw ++ ) {
            if (depth_render[ih * w + iw] > 1e5) depth_render[ih * w + iw] = 0;
        }
    }
    
    free(points3d);
    free(points3d_after);
    free(points3d_polar);
}
    
    
}//extern "C"