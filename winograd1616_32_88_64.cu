#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <random>

#define SIZE  (16*16*32)
#define OUTSIZE  (8*8*64)
#define PSIZE  (18*18*32)
#define POUTSIZE  (10*10*64)
#define FSIZE  (3*3*32*64)
#define WSIZE  (4*4*32*64)

void initialData( signed char *a, int size){
    for (int i = 0; i < size; i++) a[i] =  signed char(i);
    return;
}

__device__  signed char clamp(int v)
{
	if(v <= -128) return -128;
	else if(v > 127) return 127;
    else return v;
}

__global__ void conv( signed char *input,  signed char *filter,  signed char *output){
    // input = (16,16,32), output =(8,8,64), filter =(3,3,32,64), s_input = (18,18,32)
    
    const int id = threadIdx.x, conv_ch = blockIdx.x;

	__shared__ signed char s_input[10368];  //18*18*32
	__shared__ signed char s_filter[288];   //3*3*32
    __shared__ int s_output[64];  //8*8

    // init shared memory
    for(int i = id; i < 10368; i+=blockDim.x) s_input[i] = 0;
    for(int j = id; j < 288; j+=blockDim.x) s_filter[j] = filter[j + conv_ch*288];
    for(int k = id; k < 64; k+=blockDim.x) s_output[k] = 0;

    for(int i = id; i < 8192; i+=blockDim.x){
        const int x = i&15;
        const int y = (i&255)>>4;
        const int ch = i>>8;
        s_input[(x+1) + 18*(y+1) + 324*ch] = input[i];
    }
    __syncthreads();
    for (int n = id; n < 2048; n += blockDim.x){   //4096 = output_2d * input_ch
        const int x = n&7;
        const int y = (n&63)>>3;
        const int ch = n>>6;
        const int x0 = s_input[((x<<1))  +18*((y<<1))  + ch*324] * s_filter[0 + ch*9];
        const int x1 = s_input[((x<<1)+1)+18*((y<<1))  + ch*324] * s_filter[1 + ch*9];
        const int x2 = s_input[((x<<1)+2)+18*((y<<1))  + ch*324] * s_filter[2 + ch*9];
        const int x3 = s_input[((x<<1))  +18*((y<<1)+1)+ ch*324] * s_filter[3 + ch*9];
        const int x4 = s_input[((x<<1)+1)+18*((y<<1)+1)+ ch*324] * s_filter[4 + ch*9];
        const int x5 = s_input[((x<<1)+2)+18*((y<<1)+1)+ ch*324] * s_filter[5 + ch*9];
        const int x6 = s_input[((x<<1))  +18*((y<<1)+2)+ ch*324] * s_filter[6 + ch*9];
        const int x7 = s_input[((x<<1)+1)+18*((y<<1)+2)+ ch*324] * s_filter[7 + ch*9];
        const int x8 = s_input[((x<<1)+2)+18*((y<<1)+2)+ ch*324] * s_filter[8 + ch*9];
        atomicAdd(&s_output[x+(y<<3)], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    for (int i = id; i < 64; i+=blockDim.x) output[i + (conv_ch<<6)] = clamp(((s_output[i] + (1 << 4)) >>5)) + 128;
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char  *output){
	// dim3(8/2, 8/2) dim3(4,4,64)
    const int id = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = bx*2 + tx + (by*2+ty)*18 + tz*324;  //324 = 18*18
	
    

	__shared__ signed char input_smem [32][4][4];
	__shared__ int output_smem [64][2][2];
	__shared__ int Btd [32][4][4];
	__shared__ int BtdB [32][4][4];
	__shared__ int AtI [64][2][4];
	__shared__ int I [64][4][4];
	
	I[tz][ty][tx] = 0;
	input_smem[tz][ty][tx] = input[in_start];
	// __syncthreads();
	switch (ty)
	{
	case 0:
		Btd [tz][ty][tx] = input_smem[tz][tx][0] - input_smem[tz][tx][2];
		break;
	case 1:
		Btd [tz][ty][tx] = input_smem[tz][tx][1] + input_smem[tz][tx][2];
		break;
	case 2:
		Btd [tz][ty][tx] = - input_smem[tz][tx][1] + input_smem[tz][tx][2];
		break;
	case 3:
		Btd [tz][ty][tx] = input_smem[tz][tx][1] - input_smem[tz][tx][3];
		break;
	}
	// __syncthreads();
	switch (tx)
	{
	case 0:
		BtdB[tz][tx][ty] = Btd[tz][ty][0] - Btd[tz][ty][2];
		break;
	case 1:
		BtdB[tz][tx][ty] = Btd[tz][ty][1] + Btd[tz][ty][2];
		break;
	case 2:
		BtdB[tz][tx][ty] = - Btd[tz][ty][1] + Btd[tz][ty][2];
		break;
	case 3:
		BtdB[tz][tx][ty] = Btd[tz][ty][1] - Btd[tz][ty][3];
		break;
	}
	__syncthreads();
	for(int i=id; i<512*64; i+=512){
        const int ch = i>>9;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}

    // __syncthreads();
    if(tx == 0 && ty == 0) {
        const int out_start1 = (bx+1) + ((by+1)*10) + ((2*tz)*100);
        const int out_start2 = (bx+1) + ((by+1)*10) + ((2*tz + 1)*100);
        output[out_start1] = clamp((((I[2*tz][0][0] + I[2*tz][0][1] + I[2*tz][0][2] + I[2*tz][1][0] + I[2*tz][1][1] + I[2*tz][1][2] + I[2*tz][2][0] + I[2*tz][2][1] + I[2*tz][2][2]) + (1 << 6)) >>7)) + 128;
        output[out_start2] = clamp((((I[2*tz + 1][0][0] + I[2*tz + 1][0][1] + I[2*tz + 1][0][2] + I[2*tz + 1][1][0] + I[2*tz + 1][1][1] + I[2*tz + 1][1][2] + I[2*tz + 1][2][0] + I[2*tz + 1][2][1] + I[2*tz + 1][2][2]) + (1 << 6)) >>7)) + 128;
    }
	
}


__global__ void padding( signed char *input,  signed char *output){
    const int id  = threadIdx.x + blockDim.x*threadIdx.y;
    const int idx = threadIdx.x;
    const int idy = threadIdx.y;
    const int ch = blockIdx.x;
    __shared__  signed char s_output[18*18];

    for(int i=id; i< 18*18; i+=blockDim.x*blockDim.y) s_output[i] = 0;
    __syncthreads();
    s_output[(idx+1) + (idy+1)*18] = input[idx + (idy<<4) + (ch<<8)];
    __syncthreads();
    for(int i=id; i< 18*18; i+=blockDim.x*blockDim.y) output[i + ch*18*18] = s_output[i];
}


int main(){
    cudaEvent_t start, stop;
    float elapsed_time_ms;
     signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));

    initialData(h_char, SIZE);

    // allocate global memory
    signed char *d_char, *d_char_out, *d_char_outp, *d_charp, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, OUTSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_outp, POUTSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_charp, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, FSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, WSIZE * sizeof( signed short) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[FSIZE];
    signed short wino[WSIZE];
    fp = fopen( "./layer3.0.conv1.weight", "rb" );
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


    fp = fopen( "./wino_params_short/layer3.0.conv1.weight", "rb" );
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
    
    
    //Measure load store uint8
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i<1000; i++) {
        // cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
        conv<<<64, 256>>>(d_char, d_filter, d_char_out);
    }
    elapsed_time_ms=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms);

    signed char res[OUTSIZE];
    cudaMemcpy(res, d_char_out, sizeof(signed char) * OUTSIZE, cudaMemcpyDeviceToHost);
    for(int i=0; i<32; i++) printf("%d, ", res[i]);
    printf("\n");

    //Measure load store uint8
    padding<<<32, dim3(16,16)>>>(d_char, d_charp);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i=0; i<1000; i++){
        cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
		winograd<<<dim3(8, 8), dim3(4,4,32)>>>(d_charp, d_wino, d_char_outp);
    } 
    elapsed_time_ms=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms);
    
    signed char res1[POUTSIZE];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * POUTSIZE, cudaMemcpyDeviceToHost);
    for(int i=11; i<32; i++) printf("%d, ", char (res1[i]));
    printf("\n");


    free(h_char );
    cudaFree(d_char);
    cudaFree(d_char_out);
    cudaFree(d_char_outp);
    cudaFree(d_charp);
    cudaFree(d_filter);
    cudaFree(d_wino);

    return;
    
}