#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#define SIZE  (32*32*16)
#define PSIZE  (34*34*16)

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
	// dim3(16) dim3(4,4,16)
    const int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z, bx = blockIdx.x;
    // const int id = threadIdx.x + (threadIdx.y<<2) + (threadIdx.z<<4);
    const int tx_ty = threadIdx.x + (threadIdx.y<<2);
	__shared__ signed char input_smem[16][16][16];
	__shared__ int BtdB[16][16][16];
	__shared__ int I[16][16][4][4];
	
    // input to smem
    for(int i=0; i<16; i++){
        const int in_start = (i<<1)+tx + ((bx<<1)+ty)*34 + tz*1156;
        input_smem[i][tz][tx_ty] = input[in_start];
        I[i][tz][ty][tx] = 0;
    }
    __syncthreads();

    BtdB[tz][tx_ty][0] = input_smem[tz][tx_ty][0]-input_smem[tz][tx_ty][8]-input_smem[tz][tx_ty][2]+input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][1] = input_smem[tz][tx_ty][1]-input_smem[tz][tx_ty][9]+input_smem[tz][tx_ty][2]-input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][2] = -input_smem[tz][tx_ty][1]+input_smem[tz][tx_ty][9]+input_smem[tz][tx_ty][2]-input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][3] = input_smem[tz][tx_ty][1]-input_smem[tz][tx_ty][9]-input_smem[tz][tx_ty][3]+input_smem[tz][tx_ty][11];
    BtdB[tz][tx_ty][4] = input_smem[tz][tx_ty][4]+input_smem[tz][tx_ty][8]-input_smem[tz][tx_ty][6]-input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][5] = input_smem[tz][tx_ty][5]+input_smem[tz][tx_ty][9]+input_smem[tz][tx_ty][6]+input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][6] = -input_smem[tz][tx_ty][5]-input_smem[tz][tx_ty][9]+input_smem[tz][tx_ty][6]+input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][7] = input_smem[tz][tx_ty][5]+input_smem[tz][tx_ty][9]-input_smem[tz][tx_ty][7]-input_smem[tz][tx_ty][11];
    BtdB[tz][tx_ty][8] = -input_smem[tz][tx_ty][4]+input_smem[tz][tx_ty][8]+input_smem[tz][tx_ty][6]-input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][9] = -input_smem[tz][tx_ty][5]+input_smem[tz][tx_ty][9]-input_smem[tz][tx_ty][6]+input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][10] = input_smem[tz][tx_ty][5]-input_smem[tz][tx_ty][9]-input_smem[tz][tx_ty][6]+input_smem[tz][tx_ty][10];
    BtdB[tz][tx_ty][11] = -input_smem[tz][tx_ty][5]+input_smem[tz][tx_ty][9]+input_smem[tz][tx_ty][7]-input_smem[tz][tx_ty][11];
    BtdB[tz][tx_ty][12] = input_smem[tz][tx_ty][4]-input_smem[tz][tx_ty][12]-input_smem[tz][tx_ty][6]+input_smem[tz][tx_ty][14];
    BtdB[tz][tx_ty][13] = input_smem[tz][tx_ty][5]-input_smem[tz][tx_ty][13]+input_smem[tz][tx_ty][6]-input_smem[tz][tx_ty][14];
    BtdB[tz][tx_ty][14] = -input_smem[tz][tx_ty][5]+input_smem[tz][tx_ty][13]+input_smem[tz][tx_ty][6]-input_smem[tz][tx_ty][14];
    BtdB[tz][tx_ty][15] = input_smem[tz][tx_ty][5]-input_smem[tz][tx_ty][13]-input_smem[tz][tx_ty][7]+input_smem[tz][tx_ty][15];
    __syncthreads();

    for(int i=0; i<16; i++){
        for(int k=0; k<16; k++){
            atomicAdd(&I[i][k][ty][tx], BtdB[k][tz][tx_ty]*weight[tx_ty+(tz<<4)+(i<<8)]);
        }
    }
    // for(int i=id; i<65536; i+=256){
        // int ch = i>>12, b_x = (i&4095)>>8;
        // atomicAdd(&I[ch][b_x][ty][tx], BtdB[b_x][tz][tx_ty]*weight[tx_ty+(tz<<4)+(ch<<8)]);
    // }
    __syncthreads();
    // if(bx==0&&tz==0) printf("%d ",I[0][0][ty][tx]);
    const int out_start1 = ((tx_ty<<1)+1) + (((bx<<1)+1)*34) + ((tz)*1156);
    const int out_start2 = ((tx_ty<<1)+2) + (((bx<<1)+1)*34) + ((tz)*1156);
    const int out_start3 = ((tx_ty<<1)+1) + (((bx<<1)+2)*34) + ((tz)*1156);
    const int out_start4 = ((tx_ty<<1)+2) + (((bx<<1)+2)*34) + ((tz)*1156);
    output[out_start1] = clamp((((I[tz][tx_ty][0][0] + I[tz][tx_ty][0][1] + I[tz][tx_ty][0][2] + I[tz][tx_ty][1][0] + I[tz][tx_ty][1][1] + I[tz][tx_ty][1][2] + I[tz][tx_ty][2][0] + I[tz][tx_ty][2][1] + I[tz][tx_ty][2][2]) + (1 << 6)) >>7)) + 128;
    output[out_start2] = clamp((((I[tz][tx_ty][0][1] - I[tz][tx_ty][0][2] - I[tz][tx_ty][0][3] + I[tz][tx_ty][1][1] - I[tz][tx_ty][1][2] - I[tz][tx_ty][1][3] + I[tz][tx_ty][2][1] - I[tz][tx_ty][2][2] - I[tz][tx_ty][2][3]) + (1 << 6)) >>7)) + 128;
    output[out_start3] = clamp((((I[tz][tx_ty][1][0] + I[tz][tx_ty][1][1] + I[tz][tx_ty][1][2] - I[tz][tx_ty][2][0] - I[tz][tx_ty][2][1] - I[tz][tx_ty][2][2] - I[tz][tx_ty][3][0] - I[tz][tx_ty][3][1] - I[tz][tx_ty][3][2]) + (1 << 6)) >>7)) + 128;
    output[out_start4] = clamp((((I[tz][tx_ty][1][1] - I[tz][tx_ty][1][2] - I[tz][tx_ty][1][3] - I[tz][tx_ty][2][1] + I[tz][tx_ty][2][2] + I[tz][tx_ty][2][3] - I[tz][tx_ty][3][1] + I[tz][tx_ty][3][2] + I[tz][tx_ty][3][3]) + (1 << 6)) >>7)) + 128;

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
    
    //Measure
    padding<<<16, dim3(32,32)>>>(d_char, d_charp);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cudaMemset(&d_char_outp, 0, sizeof(signed char)*(34*34*16));
    winograd<<<dim3(16), dim3(4,4,16)>>>(d_charp, d_wino, d_char_outp);
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
    conv<<<16*2, 256>>>(d_char, d_filter, d_char_out);
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);



    //check result
    signed char res[32*32*16];
    cudaMemcpy(res, d_char_out, sizeof( signed char) * 32*32*16, cudaMemcpyDeviceToHost);
    signed char res1[34*34*16];
    cudaMemcpy(res1, d_char_outp, sizeof(signed char) * 34*34*16, cudaMemcpyDeviceToHost);

    signed char resp[PSIZE] = {0};
    for(int i=0;i<16;i++){
        for (int j=0;j<32; j++){
            for (int k=0;k<32; k++){
                resp[j+1 + (k+1)*34 + i*1156] = res[j + k*32 + i*1024];
            }
        }
    }
    for(int i=34;i<50; i++) printf("%d ", resp[i]);
    printf("\n");
    for(int i=34;i<50; i++) printf("%d ", res1[i]);
    printf("\n");

    int miss = 0;
    // for(int i=0;i<PSIZE; i++) if(resp[i] != res1[i]) {printf("%d ", i); miss++;}
    for(int i=0;i<PSIZE; i++) if(resp[i] != res1[i]) {miss++;}
    if(miss == 0) printf("%f 倍速くなりました。", elapsed_time_ms1/elapsed_time_ms2);
    else if(miss != 0) printf("bat! miss=%d", miss);

    return 0;
    
}