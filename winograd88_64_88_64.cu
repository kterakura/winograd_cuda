﻿#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define SIZE  (8*8*64)
#define PSIZE  (10*10*64)
#define FSIZE  (3*3*64*64)
#define WSIZE  (4*4*64*64)

void initialData( signed char *a, int size){
    for (int i = 0; i < size; i++) a[i] =  i;
    return;
}

__device__  signed char clamp(int v)
{
	if(v <= -128) return -128;
	else if(v > 127) return 127;
    else return v;
}





__global__ void conv( signed char *input,  signed char *filter,  signed char *output){
    // input = (8,8,64), output =(8,8,64), filter =(3,3,64,64), s_input = (10,10,64)
    
    const int id = threadIdx.x, conv_ch = blockIdx.x;

	__shared__ signed char s_input[6400];  //10*10*64
	__shared__ signed char s_filter[576];   //3*3*64
    __shared__ int s_output[64];  //8*8

    // init shared memory
    // for(int i = id; i < 6400; i+=blockDim.x) s_input[i] = 0;
    for(int j = id; j < 576; j+=blockDim.x) s_filter[j] = filter[j + conv_ch*576];
    for(int k = id; k < 64; k+=blockDim.x) s_output[k] = 0;

    // for(int i = id; i < 4096; i+=blockDim.x){
    //     const int x = i&7;
    //     const int y = (i&63)>>3;
    //     const int ch = i>>6;
    //     s_input[(x+1) + 10*(y+1) + 100*ch] = input[i];
    // }
    for(int i = id; i < 6400; i+=blockDim.x){
        s_input[i] = input[i];
    }
    __syncthreads();


    for (int n = id; n < 4096; n += blockDim.x){
        const int x = n&7;
        const int y = (n&63)>>3;
        const int ch = n>>6;
        const int x0 = s_input[(x)  +10*(y)  + ch*100] * s_filter[0 + ch*9];
        const int x1 = s_input[(x+1)+10*(y)  + ch*100] * s_filter[1 + ch*9];
        const int x2 = s_input[(x+2)+10*(y)  + ch*100] * s_filter[2 + ch*9];
        const int x3 = s_input[(x)  +10*(y+1)+ ch*100] * s_filter[3 + ch*9];
        const int x4 = s_input[(x+1)+10*(y+1)+ ch*100] * s_filter[4 + ch*9];
        const int x5 = s_input[(x+2)+10*(y+1)+ ch*100] * s_filter[5 + ch*9];
        const int x6 = s_input[(x)  +10*(y+2)+ ch*100] * s_filter[6 + ch*9];
        const int x7 = s_input[(x+1)+10*(y+2)+ ch*100] * s_filter[7 + ch*9];
        const int x8 = s_input[(x+2)+10*(y+2)+ ch*100] * s_filter[8 + ch*9];
        atomicAdd(&s_output[x+(y<<3)], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    for (int i = id; i < 64; i+=blockDim.x){
        const int x = (i&7), y = (i>>3);
        output[x+1 + ((y+1)*10) + (conv_ch*100)] = clamp(((s_output[i] + (1 << 4)) >>5)) + 128;
    } 
}

__global__ void conv_same_tiling( signed char *input,  signed char *filter,  signed char *output){
    __shared__ int s_output[64][2][2];
    __shared__ signed char input_smem [64][4][4];
    const int tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;
    for(int i=tx; i<1024; i+=1024){
        const int x = tx&3, y = (tx&15)>>2, z = tx>>4;
        const int in_start = (bx<<1)+x + ((by<<1)+y)*10 + z*100;
        input_smem[z][y][x] = input[in_start];
    }

    if(tx < 256) {
        const int x = tx&1, y = (tx&3)>>1, z = tx>>2;
        s_output[z][y][x] = 0;
    }

    __syncthreads();
    for(int i=tx; i<16384; i+=1024){  //16384 = 4*64*64
        const int x = i&1, y = (i&3)>>1, z = (i&255)>>2, ch = i>>8;
        const int f = z*9 + ch*576;
        const int x0 = input_smem[z][y+0][x+0] * filter[0 + f]; //576 = 9*64
        const int x1 = input_smem[z][y+0][x+1] * filter[1 + f];
        const int x2 = input_smem[z][y+0][x+2] * filter[2 + f];
        const int x3 = input_smem[z][y+1][x+0] * filter[3 + f];
        const int x4 = input_smem[z][y+1][x+1] * filter[4 + f];
        const int x5 = input_smem[z][y+1][x+2] * filter[5 + f];
        const int x6 = input_smem[z][y+2][x+0] * filter[6 + f];
        const int x7 = input_smem[z][y+2][x+1] * filter[7 + f];
        const int x8 = input_smem[z][y+2][x+2] * filter[8 + f];
        atomicAdd(&s_output[ch][y][x], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    
    __syncthreads();
    if(tx < 256){
        const int x = tx&1, y = (tx&3)>>1, z = tx>>2;
        const int out_start = (bx<<1)+x+1 + ((by<<1)+y+1)*10 + (z*100); 
        output[out_start] = clamp(((s_output[z][y][x] + (1 << 4)) >>5)) + 128;   
    }
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char  *output){
	// dim3(8/2, 8/2) dim3(4,4,64)
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = (bx<<1) + tx + ((by<<1)+ty)*10 + tz*100;  //100 = 10*10
    const int x_y = tx + (ty<<2);
    const int id = x_y + (threadIdx.z<<4);
	__shared__ signed char input_smem [64][16];
	__shared__ int BtdB [64][16];
	__shared__ int I [64][4][4];
	
	I[tz][ty][tx] = 0;
	input_smem[tz][x_y] = input[in_start];
    if(id < 64){
        BtdB[id][0] = input_smem[id][0]-input_smem[id][8]-input_smem[id][2]+input_smem[id][10];
        BtdB[id][1] = input_smem[id][1]-input_smem[id][9]+input_smem[id][2]-input_smem[id][10];
        BtdB[id][2] = -input_smem[id][1]+input_smem[id][9]+input_smem[id][2]-input_smem[id][10];
        BtdB[id][3] = input_smem[id][1]-input_smem[id][9]-input_smem[id][3]+input_smem[id][11];
        BtdB[id][4] = input_smem[id][4]+input_smem[id][8]-input_smem[id][6]-input_smem[id][10];
        BtdB[id][5] = input_smem[id][5]+input_smem[id][9]+input_smem[id][6]+input_smem[id][10];
        BtdB[id][6] = -input_smem[id][5]-input_smem[id][9]+input_smem[id][6]+input_smem[id][10];
        BtdB[id][7] = input_smem[id][5]+input_smem[id][9]-input_smem[id][7]-input_smem[id][11];
        BtdB[id][8] = -input_smem[id][4]+input_smem[id][8]+input_smem[id][6]-input_smem[id][10];
        BtdB[id][9] = -input_smem[id][5]+input_smem[id][9]-input_smem[id][6]+input_smem[id][10];
        BtdB[id][10] = input_smem[id][5]-input_smem[id][9]-input_smem[id][6]+input_smem[id][10];
        BtdB[id][11] = -input_smem[id][5]+input_smem[id][9]+input_smem[id][7]-input_smem[id][11];
        BtdB[id][12] = input_smem[id][4]-input_smem[id][12]-input_smem[id][6]+input_smem[id][14];
        BtdB[id][13] = input_smem[id][5]-input_smem[id][13]+input_smem[id][6]-input_smem[id][14];
        BtdB[id][14] = -input_smem[id][5]+input_smem[id][13]+input_smem[id][6]-input_smem[id][14];
        BtdB[id][15] = input_smem[id][5]-input_smem[id][13]-input_smem[id][7]+input_smem[id][15];
    }

    for(int i=0; i<64; i++){
        atomicAdd(&I[i][ty][tx], BtdB[tz][x_y]*weight[id + (i<<10)]);
    }
    __syncthreads();
    // const int temp = tx + (ty<<2);
    if(id < 64) {
        const int out_start1 = (bx*2+1) + ((by*2+1)*10) + ((id)*100);
        const int out_start2 = (bx*2+2) + ((by*2+1)*10) + ((id)*100);
        const int out_start3 = (bx*2+1) + ((by*2+2)*10) + ((id)*100);
        const int out_start4 = (bx*2+2) + ((by*2+2)*10) + ((id)*100);
        output[out_start1] = clamp((((I[id][0][0] + I[id][0][1] + I[id][0][2] + I[id][1][0] + I[id][1][1] + I[id][1][2] + I[id][2][0] + I[id][2][1] + I[id][2][2]) + (1 << 6)) >>7)) + 128;
        output[out_start2] = clamp((((I[id][0][1] - I[id][0][2] - I[id][0][3] + I[id][1][1] - I[id][1][2] - I[id][1][3] + I[id][2][1] - I[id][2][2] - I[id][2][3]) + (1 << 6)) >>7)) + 128;
        output[out_start3] = clamp((((I[id][1][0] + I[id][1][1] + I[id][1][2] - I[id][2][0] - I[id][2][1] - I[id][2][2] - I[id][3][0] - I[id][3][1] - I[id][3][2]) + (1 << 6)) >>7)) + 128;
        output[out_start4] = clamp((((I[id][1][1] - I[id][1][2] - I[id][1][3] - I[id][2][1] + I[id][2][2] + I[id][2][3] - I[id][3][1] + I[id][3][2] + I[id][3][3]) + (1 << 6)) >>7)) + 128;
    }
}


__global__ void padding( signed char *input,  signed char *output){
    const int id  = threadIdx.x + blockDim.x*threadIdx.y;
    const int idx = threadIdx.x;
    const int idy = threadIdx.y;
    const int ch = blockIdx.x;
    __shared__  signed char s_output[10*10];

    for(int i=id; i< 10*10; i+=blockDim.x*blockDim.y) s_output[i] = 0;
    __syncthreads();
    s_output[(idx+1) + (idy+1)*10] = input[idx + (idy<<3) + (ch<<6)];
    __syncthreads();
    for(int i=id; i< 10*10; i+=blockDim.x*blockDim.y) output[i + ch*10*10] = s_output[i];
}


int main(){
    cudaEvent_t start, stop;
    float elapsed_time_ms1, elapsed_time_ms2, elapsed_time_ms3;
    signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));

    initialData(h_char, SIZE);

    // allocate global memory
    signed char *d_char, *d_char_out, *d_char_outp, *d_char_outp1, *d_charp, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_outp, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_outp1, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_charp, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, FSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, WSIZE * sizeof( signed short) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[FSIZE];
    signed short wino[WSIZE];
    fp = fopen( "./params/layer3.0.conv2.weight", "rb" );
    if (!fp) printf("x1_1: pathを間違えています\n");
    for(int i=0; i<FSIZE; i++){
        if( fread( &f, sizeof(f), 1, fp ) < 1 ){
            fputs( "x1_1: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        x1_1[i] = f;
    }
    cudaMemcpy(d_filter, x1_1, sizeof( signed char) * FSIZE, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);


    fp = fopen( "./wino_params_short/layer3.0.conv2.weight", "rb" );
    if (!fp) printf("wino: pathを間違えています\n");
    for(int i=0; i<WSIZE; i++){
        if( fread( &f_short, sizeof(f_short), 1, fp ) < 1 ){
            fputs( "wino: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        wino[i] = f_short;
    }
    cudaMemcpy(d_wino, wino, sizeof(signed short) * WSIZE, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);
    
    padding<<<64, dim3(8,8)>>>(d_char, d_charp);
    
    //Measure
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemset(&d_char_outp1, 0, sizeof(signed char)*PSIZE);
    winograd<<<dim3(4, 4), dim3(4,4,64)>>>(d_charp, d_wino, d_char_outp1);
    elapsed_time_ms2=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms2, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms2);
    

    //Measure
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
    conv_same_tiling<<<dim3(4, 4), 1024>>>(d_charp, d_filter, d_char_outp);
    elapsed_time_ms3=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms3, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("same_tiling:%f\n", elapsed_time_ms3);
    
    signed char res2[PSIZE];
    cudaMemcpy(res2, d_char_outp, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);

    //Measure
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
    conv<<<64, 256>>>(d_charp, d_filter, d_char_outp);
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);


    signed char res1[PSIZE];
    cudaMemcpy(res1, d_char_outp1, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);
    signed char res[PSIZE];
    cudaMemcpy(res, d_char_outp, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);

    int miss = 0;
    for(int i=0;i<PSIZE; i++) if(res[i] != res1[i] && res[i] != res2[i]) {printf("%d ", i); miss++;}
    if(miss == 0) printf("%f 倍速くなりました。normal/wino\n", elapsed_time_ms1/elapsed_time_ms2);
    if(miss == 0) printf("%f 倍速くなりました。same_tiling/wino\n", elapsed_time_ms3/elapsed_time_ms2);
    if(miss == 0) printf("%f 倍速くなりました。normal/same_tiling\n", elapsed_time_ms1/elapsed_time_ms3);
    else if(miss != 0) printf("bat!");

    free(h_char );
    cudaFree(d_char);
    cudaFree(d_char_out);
    cudaFree(d_char_outp);
    cudaFree(d_char_outp1);
    cudaFree(d_charp);
    cudaFree(d_filter);
    cudaFree(d_wino);

    return 0;
    
}