#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <random>

#define SIZE  (32*32*16)
#define PSIZE  (34*34*16)

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
    __shared__ int s_output[16];
    __shared__ signed char input_smem [16][9];
    const int tz = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;
    for(int i=0; i<9; i++){
        const int x = i%3, y = i/3;
        const int in_start = bx+x + (by+y)*34 + tz*1156;  //1156 = 34*34
        input_smem[tz][i] = input[in_start];
    }
    
    s_output[tz] = 0;
    __syncthreads();
    for(int i=0; i<16; i++){
        const int x0 = input_smem[tz][0] * filter[0 + tz*9 + i*144]; //144 = 9*16
        const int x1 = input_smem[tz][1] * filter[1 + tz*9 + i*144];
        const int x2 = input_smem[tz][2] * filter[2 + tz*9 + i*144];
        const int x3 = input_smem[tz][3] * filter[3 + tz*9 + i*144];
        const int x4 = input_smem[tz][4] * filter[4 + tz*9 + i*144];
        const int x5 = input_smem[tz][5] * filter[5 + tz*9 + i*144];
        const int x6 = input_smem[tz][6] * filter[6 + tz*9 + i*144];
        const int x7 = input_smem[tz][7] * filter[7 + tz*9 + i*144];
        const int x8 = input_smem[tz][8] * filter[8 + tz*9 + i*144];
        atomicAdd(&s_output[i], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    const int out_start = bx+1 + (by+1)*34 + (tz*1156);
    output[out_start] = clamp(((s_output[tz] + (1 << 4)) >>5)) + 128;
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char *output){
	// dim3(32/2, 32/2) dim3(4,4,16)
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = bx*2 + tx + (by*2+ty)*34 + tz*1156;  //1156 = 34*34

	// dim3(32/2, 32/2, 16) dim3(16,4,4)
	// const int in_start = tx + ((ty + (bx<<1))<<4) + (tz + (by<<1))*544;  //1156 = 34*34


	__shared__ signed char input_smem [16][4][4];
	__shared__ int Btd [16][4][4];
	__shared__ int BtdB [16][4][4];
	__shared__ int I [16][4][4];
	
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
	// __syncthreads();
    const int id = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
	for(int i=id; i<4*4*16*16; i+=blockDim.x*blockDim.y*blockDim.z){
        const int ch = i>>8;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}
    __syncthreads();
    if(id < 16) {
        const int out_start1 = (bx*2+1) + ((by*2+1)*34) + ((id)*1156);
        const int out_start2 = (bx*2+2) + ((by*2+1)*34) + ((id)*1156);
        const int out_start3 = (bx*2+1) + ((by*2+2)*34) + ((id)*1156);
        const int out_start4 = (bx*2+2) + ((by*2+2)*34) + ((id)*1156);
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
    __shared__  signed char s_output[34*34];

    for(int i=id; i< 34*34; i+=blockDim.x*blockDim.y) s_output[i] = 0;
    __syncthreads();
    s_output[(idx+1) + (idy+1)*34] = input[idx + idy*32 + ch*32*32];
    __syncthreads();
    for(int i=id; i< 34*34; i+=blockDim.x*blockDim.y) output[i + ch*34*34] = s_output[i];
}


int main(){
    cudaEvent_t start, stop;
    float elapsed_time_ms1, elapsed_time_ms2;
    signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));
    initialData(h_char, SIZE);

    // allocate global memory
    signed char *d_char, *d_char_out, *d_char_outp, *d_charp, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_outp, 34*34*16 * sizeof( signed char) );
    cudaMalloc( (void **) &d_charp, 34*34*16 * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, 3*3*16*16 * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, 4*4*16*16 * sizeof( signed short ) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[16*16*3*3];
    signed short wino[16*16*4*4];
    fp = fopen( "./params/layer1.0.conv1.weight", "rb" );
    if (!fp) printf("x1_1: pathを間違えています\n");
    for(int i=0; i<16*16*3*3; i++){
        if( fread( &f, sizeof(f), 1, fp ) < 1 ){
            fputs( "x1_1: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        x1_1[i] = f;
    }
    cudaMemcpy(d_filter, x1_1, sizeof( signed char) * 16*16*3*3, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);


    fp = fopen( "./wino_params_short/layer1.0.conv1.weight", "rb" );
    if (!fp) printf("wino: pathを間違えています\n");
    for(int i=0; i<16*16*4*4; i++){
        if( fread( &f_short, sizeof(f_short), 1, fp ) < 1 ){
            fputs( "wino: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        wino[i] = f_short;
    }
    cudaMemcpy(d_wino, wino, sizeof(signed short) * 16*16*4*4, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);
    

    padding<<<16, dim3(32,32)>>>(d_char, d_charp);
    //Measure load store uint8
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i<1000; i++) {
        cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
        conv<<<dim3(32, 32), dim3(16)>>>(d_charp, d_filter, d_char_outp);
    }
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);

    signed char res[PSIZE];
    cudaMemcpy(res, d_char_outp, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);
    

    //Measure load store uint8
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i<1000; i++){
        cudaMemset(&d_char_outp, 0, sizeof(signed char)*(34*34*16));
		winograd<<<dim3(16, 16), dim3(4,4,16)>>>(d_charp, d_wino, d_char_outp);
    } 
    elapsed_time_ms2=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms2, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms2);
    
    signed char res1[34*34*16];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * 34*34*16, cudaMemcpyDeviceToHost);

    //check data
    int miss = 0;
    for(int i=0;i<PSIZE; i++) if(res[i] != res1[i]) {printf("%d ", i); miss++;}
    if(miss == 0) printf("%f 倍速くなりました。", elapsed_time_ms1/elapsed_time_ms2);
    else if(miss != 0) printf("%f 倍速くなりました。答え一致してないけどね", elapsed_time_ms1/elapsed_time_ms2);
    return;

    return;
    
}