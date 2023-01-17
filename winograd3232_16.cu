#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>
#include <windows.h>
#include <iostream>
#include <random>

#define SIZE  (32*32*16)

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
    // input = (32,16,16), output =(32,16,16), filter =(3,3,16,16), s_input = (34,18,16)

    const int id = threadIdx.x + blockDim.x*threadIdx.y, ch_id = blockIdx.x, block_size = blockDim.x*blockDim.y;
    const int conv_place = (ch_id&1), conv_ch = (ch_id>>1);
    
    __shared__  signed char s_input[9792]; //34*18*16
	__shared__  signed char s_filter[144]; //3*3*16
    __shared__ int s_output[512];  //32*16
    
    // init shared memory
    for(int i = id; i < 9792; i+=block_size) s_input[i] = 0;
    for(int j = id; j < 144; j+=block_size) s_filter[j] = filter[j + conv_ch*144];
    for(int k = id; k < 512; k+=block_size) s_output[k] = 0;

    for (int i = id; i < 8704; i+=blockDim.x){   //8704 = 32*(16+1)*16,   544 = 32*(16+1)
        const int x = i&31, y = ((i%544)>>5), ch = i/(544);
        s_input[(x+1) + 34*(y+(conv_place^1)) + 612*ch] = input[x + ((y + (conv_place<<4) - conv_place)<<5) + (ch<<10)];
    }__syncthreads();

    for (int n = id; n < 8192; n += block_size){  //9782 = output_2d * input_ch
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
        output[x + (y<<5) + (conv_ch<<10)] = clamp(((s_output[i] + (1 << 4)) >>5)) + 128;
    }
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char *output){
	// dim3(32/2, 32/2) dim3(4,4,16)
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x, by = blockIdx.y;
	const int in_start = bx*2 + tx + (by*2+ty)*34 + tz*1156;  //1156 = 34*34
	const int out_start = (bx<<1)+tx + (((by<<1)+ty)<<5) + (tz<<10);  //1024 = 32*32

	// dim3(32/2, 32/2, 16) dim3(16,4,4)
	// const int in_start = tx + ((ty + (bx<<1))<<4) + (tz + (by<<1))*544;  //1156 = 34*34


	__shared__ signed char input_smem [16][4][4];
	__shared__ int Btd [16][4][4];
	__shared__ int BtdB [16][4][4];
	__shared__ int AtI [16][2][4];
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
        // const int z = ch>>4;
        // const int x = i&3;
        // const int y = (i&15)>>2;
		atomicAdd(&I[ch][ty][tx], BtdB[tz][ty][tx]*weight[i]);
	}
    __syncthreads();

	// for(int i=0; i<16; i++){
	// 	const int w_start = tx + (ty<<2)+ (tz<<4) + (i*16*16);  //256 = 4*4*16
	// 	atomicAdd(&I[i][ty][tx], BtdB[tz][ty][tx]*weight[w_start]);
	// }
	// __syncthreads();

	if(ty > 1) return;
	switch (ty)
	{
	case 0:
		AtI[tz][ty][tx] = I[tz][0][tx] + I[tz][1][tx] + I[tz][2][tx];
		break;
	case 1:
		AtI[tz][ty][tx] = I[tz][1][tx] - I[tz][2][tx] - I[tz][3][tx];
		break;
	}
	// __syncthreads();

	if(tx > 1) return;
	switch (tx)
	{
	case 0:
		output[out_start] = clamp((((AtI[tz][ty][0] + AtI[tz][ty][1] + AtI[tz][ty][2]) + (1 << 6)) >>7)) + 128;
		break;
	case 1:
		output[out_start] = clamp((((AtI[tz][ty][1] - AtI[tz][ty][2] - AtI[tz][ty][3]) + (1 << 6)) >>7)) + 128;
		break;
	}
	// __syncthreads();
	// output[out_start] = clamp(((output_smem[tz][ty][tx] + (1 << 4)) >>5)) + 128;
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
    float elapsed_time_ms;
    signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));
    signed char *h_filter = ( signed char *)malloc(9*16*16 * sizeof( signed char));
    signed short *h_wino = ( signed short *)malloc(16*16*16 * sizeof( signed short));

    initialData(h_char, SIZE);

	// for(int i=0;i<SIZE;i++){
	// 	if(i%32 == 0) printf("\n");
	// 	if(i%1024 == 0) printf("\n");
	// 	printf("%d ", h_char[i]);
	// }

    // allocate global memory
    signed char *d_char, *d_char_out, *d_char_out1, *d_charp, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out1, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_charp, 34*34*16 * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, 3*3*16*16 * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, 4*4*16*16 * sizeof( signed short ) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );
    cudaMemcpy( d_filter, h_filter, 9*16*16 * sizeof( signed char), cudaMemcpyHostToDevice );
    cudaMemcpy( d_wino, h_wino, 16*16*16 * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[16*16*3*3];
    signed short wino[16*16*4*4];
    fp = fopen( "layer1.0.conv1.weight", "rb" );
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
    

    //Measure load store uint8
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    for(int i=0; i<1000; i++) conv<<<16*2, 256>>>(d_char, d_filter, d_char_out);
    elapsed_time_ms=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms);

    signed char res[32*32*16];
    cudaMemcpy(res, d_char_out, sizeof( signed char) * 32*32*16, cudaMemcpyDeviceToHost);
    

    //Measure load store uint8
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // make_wino<<<16, dim3(4,4,16)>>>(d_filter, f_wino);
    // padding<<<16, dim3(32,32)>>>(d_char, d_charp);
	// winograd<<<dim3(16, 16), dim3(4,4,16)>>>(d_charp, d_wino, d_char_out1);
    for(int i=0; i<1000; i++){
        padding<<<16, dim3(32,32)>>>(d_char, d_charp);
		winograd<<<dim3(16, 16), dim3(4,4,16)>>>(d_charp, d_wino, d_char_out1);
    } 
    elapsed_time_ms=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms);
    
    signed char res1[32*32*16];
    cudaMemcpy(res1, d_char_out1, sizeof(signed char) * 32*32*16, cudaMemcpyDeviceToHost);
    for(int i=0; i<32; i++){
		for(int k=0; k<32; k++) printf("%d ", char (res[k + 32*i]));
		printf("\n");
		for(int k=0; k<32; k++) printf("%d ", char (res1[k + 32*i]));
		printf("\n");
		printf("\n");
	}printf("\n");

    return;
    
}