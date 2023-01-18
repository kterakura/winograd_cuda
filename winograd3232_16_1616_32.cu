#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#define SIZE  (32*32*16)
#define OUTSIZE  (16*16*32)
#define PSIZE  (34*34*16)
#define POUTSIZE  (18*18*32)
#define FSIZE  (3*3*16*32)
#define WSIZE  (4*4*16*32)

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
    // input = (32,16,16), output =(16,8,32), filter =(3,3,16,32), s_input = (34,18,16)

    const int id = threadIdx.x + blockDim.x*threadIdx.y, ch_id = blockIdx.x, block_size = blockDim.x*blockDim.y;
    const int conv_place = (ch_id&1), conv_ch = (ch_id>>1);
    
    __shared__ signed char s_input[9792]; //34*18*16
	__shared__ signed char s_filter[144]; //3*3*16
    __shared__ int s_output[128];  //8*16
    
    // init shared memory
    for(int i = id; i < 9792; i+=block_size) s_input[i] = 0;
    for(int j = id; j < 144; j+=block_size) s_filter[j] = filter[j + conv_ch*144];
    for(int k = id; k < 128; k+=block_size) s_output[k] = 0;

    for (int i = id; i < 8704; i+=blockDim.x){   //8704 = 32*(16+1)*16,   544 = 32*(16+1)
        const int x = i&31, y = ((i%544)>>5), ch = i/(544);
        s_input[(x+1) + 34*(y+(conv_place^1)) + 612*ch] = input[x + ((y + (conv_place<<4) - conv_place)<<5) + (ch<<10)];
    }__syncthreads();

    for (int n = id; n < 2048; n += block_size){  //2048 = output_2d * input_ch
        const int x = n&15;
        const int y = (n&127)>>4;
        const int ch = n>>7;
        const int x0 = s_input[((x<<1))  +34*((y<<1))  + 612*ch] * s_filter[0 + 9*ch];
        const int x1 = s_input[((x<<1)+1)+34*((y<<1))  + 612*ch] * s_filter[1 + 9*ch];
        const int x2 = s_input[((x<<1)+2)+34*((y<<1))  + 612*ch] * s_filter[2 + 9*ch];
        const int x3 = s_input[((x<<1))  +34*((y<<1)+1)+ 612*ch] * s_filter[3 + 9*ch];
        const int x4 = s_input[((x<<1)+1)+34*((y<<1)+1)+ 612*ch] * s_filter[4 + 9*ch];
        const int x5 = s_input[((x<<1)+2)+34*((y<<1)+1)+ 612*ch] * s_filter[5 + 9*ch];
        const int x6 = s_input[((x<<1))  +34*((y<<1)+2)+ 612*ch] * s_filter[6 + 9*ch];
        const int x7 = s_input[((x<<1)+1)+34*((y<<1)+2)+ 612*ch] * s_filter[7 + 9*ch];
        const int x8 = s_input[((x<<1)+2)+34*((y<<1)+2)+ 612*ch] * s_filter[8 + 9*ch];
        atomicAdd(&s_output[x+(y<<4)], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    for (int i = id; i < 128; i+=blockDim.x){
        const int x = (i&15), y = (i>>4) + (conv_place<<3);
        output[x + (y<<4) + (conv_ch<<8)] = clamp((s_output[i] + (1 << 4)) >> 5) + 128;
    }
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char  *output){
	// dim3(8/2, 8/2) dim3(4,4,64)
    const int id = threadIdx.x + (threadIdx.y<<2) + (threadIdx.z<<4);
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = (bx<<1) + tx + ((by<<1)+ty)*34 + tz*1156;  //1156 = 34*34
	
    

	__shared__ signed char input_smem [16][4][4];
	__shared__ int Btd [16][4][4];
	__shared__ int BtdB [16][4][4];
	__shared__ int I [32][4][4];
	
	I[2*tz][ty][tx] = 0;
	I[2*tz+1][ty][tx] = 0;
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
	for(int i=id; i<8192; i+=256){
        const int ch = i>>8;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}

    __syncthreads();
    if(id < 32) {
        const int out_start1 = (bx+1) + ((by+1)*18) + ((id)*324);
        output[out_start1] = clamp((((I[id][0][0] + I[id][0][1] + I[id][0][2] + I[id][1][0] + I[id][1][1] + I[id][1][2] + I[id][2][0] + I[id][2][1] + I[id][2][2]) + (1 << 6)) >>7)) + 128;
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
    s_output[(idx+1) + (idy+1)*34] = input[idx + (idy<<5) + (ch<<10)];
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
    fp = fopen( "./params/layer2.0.conv1.weight", "rb" );
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


    fp = fopen( "./wino_params_short/layer2.0.conv1.weight", "rb" );
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
    padding<<<16, dim3(32,32)>>>(d_char, d_charp);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMemset(&d_char_outp, 0, sizeof(signed char)*PSIZE);
    winograd<<<dim3(16, 16), dim3(4,4,16)>>>(d_charp, d_wino, d_char_outp);
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
    conv<<<64, 256>>>(d_char, d_filter, d_char_out);
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);

    signed char res[OUTSIZE];
    cudaMemcpy(res, d_char_out, sizeof(signed char) * OUTSIZE, cudaMemcpyDeviceToHost);
    signed char res1[POUTSIZE];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * POUTSIZE, cudaMemcpyDeviceToHost);

    //check result
    signed char resp[POUTSIZE] = {0};
    for(int i=0;i<32;i++){
        for (int j=0;j<16; j++){
            for (int k=0;k<16; k++){
                resp[j+1 + (k+1)*18 + i*324] = res[j + k*16 + i*256];
            }
        }
    }

    int miss = 0;
    for(int i=0;i<POUTSIZE; i++) if(resp[i] != res1[i]) {printf("%d ", i); miss++;}
    // for(int i=0;i<POUTSIZE; i++) if(resp[i] != res1[i]) {miss++;}
    if(miss == 0) printf("%f 倍速くなりました。", elapsed_time_ms1/elapsed_time_ms2);
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