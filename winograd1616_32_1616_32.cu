#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>

#define SIZE  (16*16*32)
#define PSIZE  (18*18*32)
#define FSIZE  (3*3*32*32)
#define WSIZE  (4*4*32*32)

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
    const int id = threadIdx.x, conv_ch = blockIdx.x;

	__shared__ signed char s_input[10368];  //18*18*32
	__shared__ signed char s_filter[288];   //3*3*32
    __shared__ int s_output[256];  //16*16

    // init shared memory
    for(int i = id; i < 10368; i+=blockDim.x) s_input[i] = 0;
    for(int j = id; j < 288; j+=blockDim.x) s_filter[j] = filter[j + conv_ch*288];
    for(int k = id; k < 256; k+=blockDim.x) s_output[k] = 0;

    for(int i = id; i < 8192; i+=blockDim.x){
        const int x = i&15;
        const int y = (i&255)>>4;
        const int ch = i>>8;
        s_input[(x+1) + 18*(y+1) + 324*ch] = input[i];
    }
    __syncthreads();

    for (int n = id; n < 8192; n += blockDim.x){
        const int x = n&15;
        const int y = (n&255)>>4;
        const int ch = n>>8;
        const int x0 = s_input[(x)  +18*(y)  + ch*324] * s_filter[0 + ch*9];
        const int x1 = s_input[(x+1)+18*(y)  + ch*324] * s_filter[1 + ch*9];
        const int x2 = s_input[(x+2)+18*(y)  + ch*324] * s_filter[2 + ch*9];
        const int x3 = s_input[(x)  +18*(y+1)+ ch*324] * s_filter[3 + ch*9];
        const int x4 = s_input[(x+1)+18*(y+1)+ ch*324] * s_filter[4 + ch*9];
        const int x5 = s_input[(x+2)+18*(y+1)+ ch*324] * s_filter[5 + ch*9];
        const int x6 = s_input[(x)  +18*(y+2)+ ch*324] * s_filter[6 + ch*9];
        const int x7 = s_input[(x+1)+18*(y+2)+ ch*324] * s_filter[7 + ch*9];
        const int x8 = s_input[(x+2)+18*(y+2)+ ch*324] * s_filter[8 + ch*9];
        atomicAdd(&s_output[x+(y<<4)], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    for (int i = id; i < 256; i+=blockDim.x) output[i + (conv_ch<<8)] = clamp(((s_output[i] + (1 << 4)) >>5)) + 128;
}

__global__ void conv_same_tiling( signed char *input,  signed char *filter,  signed char *output){
    __shared__ int s_output[32][2][2];
    __shared__ signed char input_smem [32][4][4];
    const int tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;
    for(int i=tx; i<512; i+=512){
        const int x = tx&3, y = (tx&15)>>2, z = tx>>4;
        const int in_start = (bx<<1)+x + ((by<<1)+y)*18 + z*324;
        input_smem[z][y][x] = input[in_start];
    }

    if(tx < 128) {
        const int x = tx&1, y = (tx&3)>>1, z = tx>>2;
        s_output[z][y][x] = 0;
    }

    __syncthreads();
    for(int i=tx; i<4096; i+=512){  //4096 = 4*32*32
        const int x = i&1, y = (i&3)>>1, z = (i&127)>>2, ch = i>>7;
        const int f = z*9 + ch*288;
        const int x0 = input_smem[z][y+0][x+0] * filter[0 + f]; //288 = 9*32
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
    if(tx < 128){
        const int x = tx&1, y = (tx&3)>>1, z = tx>>2;
        const int out_start = (bx<<1)+x+1 + ((by<<1)+y+1)*18 + (z*324); 
        output[out_start] = clamp(((s_output[z][y][x] + (1 << 4)) >>5)) + 128;   
    }
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char  *output){
	// dim3(32/2, 32/2) dim3(4,4,16)
    const int id = threadIdx.x + (threadIdx.y<<2) + (threadIdx.z<<4);
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = (bx<<1) + tx + ((by<<1)+ty)*18 + tz*324;  //324 = 18*18

	// dim3(32/2, 32/2, 16) dim3(16,4,4)
	// const int in_start = tx + ((ty + (bx<<1))<<4) + (tz + (by<<1))*544;


	__shared__ signed char input_smem [32][4][4];
	__shared__ int Btd [32][4][4];
	__shared__ int BtdB [32][4][4];
	__shared__ int I [32][4][4];
	
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

    for(int i=id; i<16384; i+=512){ //16384 = 4*4*32*32
        const int ch = i>>9;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}
	__syncthreads();
    if(id < 32) {
        const int out_start1 = ((bx<<1)+1) + (((by<<1)+1)*18) + ((id)*324);
        const int out_start2 = ((bx<<1)+2) + (((by<<1)+1)*18) + ((id)*324);
        const int out_start3 = ((bx<<1)+1) + (((by<<1)+2)*18) + ((id)*324);
        const int out_start4 = ((bx<<1)+2) + (((by<<1)+2)*18) + ((id)*324);
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
    __shared__  signed char s_output[18*18];

    for(int i=id; i< 18*18; i+=blockDim.x*blockDim.y) s_output[i] = 0;
    __syncthreads();
    s_output[(idx+1) + (idy+1)*18] = input[idx + (idy<<4) + (ch<<8)];
    __syncthreads();
    for(int i=id; i< 18*18; i+=blockDim.x*blockDim.y) output[i + ch*18*18] = s_output[i];
}


int main(){
    cudaEvent_t start, stop;
    float elapsed_time_ms1, elapsed_time_ms2, elapsed_time_ms3;
    signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));

    initialData(h_char, SIZE);
    // allocate global memory
    signed char *d_char, *d_char_out, *d_char_outp, *d_charp, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_outp, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_charp, PSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, FSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, WSIZE * sizeof( signed short) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[FSIZE];
    signed short wino[WSIZE];
    fp = fopen( "./params/layer2.0.conv2.weight", "rb" );
    if (!fp) printf("x1_1: pathを間違えています\n");
    for(int i=0; i<FSIZE; i++){
        if( fread( &f, sizeof(f), 1, fp ) < 1 ){
            fputs( "x1_1: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        x1_1[i] = f;
    }
    cudaMemcpy(d_filter, x1_1, sizeof(signed char) * FSIZE, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);


    fp = fopen( "./wino_params_short/layer2.0.conv2.weight", "rb" );
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
    
    
    //Measure
    padding<<<32, dim3(16,16)>>>(d_char, d_charp);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
    winograd<<<dim3(8, 8), dim3(4,4,32)>>>(d_charp, d_wino, d_char_outp);
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
    conv<<<32, 256>>>(d_char, d_filter, d_char_out);
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);

    //Measure
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    conv_same_tiling<<<dim3(8, 8), 512>>>(d_charp, d_filter, d_char_outp);
    elapsed_time_ms3=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms3, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("same_tiling:%f\n", elapsed_time_ms3);
    signed char res2[PSIZE];
    cudaMemcpy(res2, d_char_outp, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);


    
    //check result
    signed char res1[PSIZE];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * PSIZE, cudaMemcpyDeviceToHost);
    signed char res[SIZE];
    cudaMemcpy(res, d_char_out, sizeof(signed char) * SIZE, cudaMemcpyDeviceToHost);

    signed char resp[PSIZE] = {0};
    for(int i=0;i<32;i++){
        for (int j=0;j<16; j++){
            for (int k=0;k<16; k++){
                resp[j+1 + (k+1)*18 + i*324] = res[j + k*16 + i*256];
            }
        }
    }

    int miss = 0;
    for(int i=0;i<PSIZE; i++) if(resp[i] != res1[i] && resp[i] != res2[i]) {printf("%d ", i); miss++;}
    if(miss == 0) printf("%f 倍速くなりました。normal/wino\n", elapsed_time_ms1/elapsed_time_ms2);
    if(miss == 0) printf("%f 倍速くなりました。same_tiling/wino\n", elapsed_time_ms3/elapsed_time_ms2);
    if(miss == 0) printf("%f 倍速くなりました。normal/same_tiling\n", elapsed_time_ms1/elapsed_time_ms3);
    else if(miss != 0) printf("bat!");

    free(h_char );
    cudaFree(d_char);
    cudaFree(d_char_out);
    cudaFree(d_char_outp);
    cudaFree(d_charp);
    cudaFree(d_filter);
    cudaFree(d_wino);

    return 0;
    
}