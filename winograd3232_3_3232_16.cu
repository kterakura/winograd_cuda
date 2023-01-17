#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <random>

#define SIZE  (32*32*3)
#define OUTSIZE  (32*32*16)
#define PSIZE  (34*34*3)
#define POUTSIZE  (34*34*16)
#define FSIZE  (3*3*3*16)
#define WSIZE  (4*4*3*16)

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
    // input = (32,16,3), output =(32,16,16), filter =(3,3,3,16), s_input = (34,18,3)

    const int id = threadIdx.x + blockDim.x*threadIdx.y, ch_id = blockIdx.x, block_size = blockDim.x*blockDim.y;
    const int conv_place = (ch_id&1), conv_ch = (ch_id>>1);
    
    __shared__ signed char s_input[1836]; //34*18*3
	__shared__ signed char s_filter[27]; //3*3*3
    __shared__ int s_output[512];  //32*16
    
    // init shared memory
    for(int i = id; i < 1836; i+=block_size) s_input[i] = 0;
    for(int j = id; j < 27; j+=block_size) s_filter[j] = filter[j + conv_ch*27];
    for(int k = id; k < 512; k+=block_size) s_output[k] = 0;

    for (int i = id; i < 1632; i+=blockDim.x){   //1632 = 32*(16+1)*3,   544 = 32*(16+1)
        const int x = i&31, y = ((i%544)>>5), ch = i/(544);
        s_input[(x+1) + 34*(y+(conv_place^1)) + 612*ch] = input[x + ((y + (conv_place<<4) - conv_place)<<5) + (ch<<10)];
    }__syncthreads();

    for (int n = id; n < 1536; n += block_size){
        const int x = n&31;
        const int y = (n&511)>>5;
        const int ch = n>>9;
        const int x0 = s_input[(x)  +34*(y)  + 612*ch] * s_filter[0 + 9*ch];
        const int x1 = s_input[(x+1)+34*(y)  + 612*ch] * s_filter[1 + 9*ch];
        const int x2 = s_input[(x+2)+34*(y)  + 612*ch] * s_filter[2 + 9*ch];
        const int x3 = s_input[(x)  +34*(y+1)+ 612*ch] * s_filter[3 + 9*ch];
        const int x4 = s_input[(x+1)+34*(y+1)+ 612*ch] * s_filter[4 + 9*ch];
        const int x5 = s_input[(x+2)+34*(y+1)+ 612*ch] * s_filter[5 + 9*ch];
        const int x6 = s_input[(x)  +34*(y+2)+ 612*ch] * s_filter[6 + 9*ch];
        const int x7 = s_input[(x+1)+34*(y+2)+ 612*ch] * s_filter[7 + 9*ch];
        const int x8 = s_input[(x+2)+34*(y+2)+ 612*ch] * s_filter[8 + 9*ch];
        atomicAdd(&s_output[x+(y<<5)], x0+x1+x2+x3+x4+x5+x6+x7+x8);
    }
    __syncthreads();
    for (int i = id; i < 512; i+=blockDim.x){
        const int x = i&31, y = (i>>5) + (conv_place<<4);
        output[x + (y<<5) + (conv_ch<<10)] = clamp(((s_output[i] + (1 << 4)) >> 5)) + 128;
    } 
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char  *output){
    const int id = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = bx*2 + tx + (by*2+ty)*34 + tz*1156;  //1156 = 34*34
	
    

	__shared__ signed char input_smem [3][4][4];
	__shared__ int Btd [3][4][4];
	__shared__ int BtdB [3][4][4];
	__shared__ int I [16][4][4];
	
	for(int i=id; i<16*4*4; i+=48){
        const int z = i>>4, y = (i&15)>>2, x = i&3;
        I[z][y][x] = 0;
    }
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

    for(int i=id; i<48*16; i+=48){
        const int ch = i/48;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}
    __syncthreads();
    
    if(id < 16){
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
    fp = fopen( "./params/conv_block.conv.weight", "rb" );
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


    fp = fopen( "./wino_params_short/conv_block.conv.weight", "rb" );
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
        conv<<<32, 256>>>(d_char, d_filter, d_char_out);
    }
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);

    signed char res[OUTSIZE];
    cudaMemcpy(res, d_char_out, sizeof(signed char) * OUTSIZE, cudaMemcpyDeviceToHost);
    

    //Measure load store uint8
    padding<<<3, dim3(32,32)>>>(d_char, d_charp);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i=0; i<1000; i++){
        cudaMemset(&d_char_outp, 0, sizeof(signed char)*POUTSIZE);
		winograd<<<dim3(16, 16), dim3(4,4,3)>>>(d_charp, d_wino, d_char_outp);
    } 
    elapsed_time_ms2=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms2, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms2);
    
    signed char res1[POUTSIZE];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * POUTSIZE, cudaMemcpyDeviceToHost);
    

    //check data
    signed char resp[POUTSIZE] = {0};
    for(int i=0;i<16;i++){
        for (int j=0;j<32; j++){
            for (int k=0;k<32; k++){
                resp[j+1 + (k+1)*34 + i*1156] = res[j + k*32 + i*1024];
            }
        }
    }

    int miss = 0;
    for(int i=0;i<POUTSIZE; i++) if(resp[i] != res1[i]) {printf("%d ", i); miss++;}
    if(miss == 0) printf("%f 倍速くなりました。", elapsed_time_ms1/elapsed_time_ms2);
    else if(miss != 0) printf("bat!");

    // for(int i=0; i<34*34; i++){
    //     if(i%34 == 0)printf("\n"); 
    //     printf("%d ", resp[i]);
    // } printf("\n"); 
    // for(int i=0; i<34*34; i++){
    //     if(i%34 == 0)printf("\n"); 
    //     printf("%d ", res1[i]);
    // } printf("\n");

    free(h_char );
    cudaFree(d_char);
    cudaFree(d_char_out);
    cudaFree(d_char_outp);
    cudaFree(d_charp);
    cudaFree(d_filter);
    cudaFree(d_wino);

    return;
    
}