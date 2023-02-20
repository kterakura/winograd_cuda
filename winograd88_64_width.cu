#include <cstdio>
#include <time.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#define IN (10)
#define INCH (64)
#define OUTCH (64)
#define SIZE (IN*IN*INCH)
#define FSIZE (INCH*INCH*3*3)
#define FSIZE_WINO (INCH*INCH*4*4)

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
    __shared__ int s_output[4][64];
    __shared__ signed char input_smem [16][64];
    const int tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;
    const int z = tx&63, x=tx>>6;
    const int start_xy= ((bx<<1)<<6) + ((by<<1)*640) + tx;
    input_smem[x   ][z] = input[start_xy];
    input_smem[x+4 ][z] = input[start_xy+640];
    input_smem[x+8 ][z] = input[start_xy+1280];
    input_smem[x+12][z] = input[start_xy+1920];
    s_output[x   ][z] = 0;
    
    // __shared__ signed char filter_smem [36864];
    // for(int i=tx;i<36864;i+=256) filter_smem[i] = filter[i];
    
    __syncthreads();
    for(int i=0;i<4;i++){
        for(int k=0;k<9;k++){
            const int t = k+(k/3)+(i&1)+((i>>1)<<2);
            atomicAdd(&s_output[i][(z+x   )&63], input_smem[t][z]*filter[tx     +(k<<12)]);
            atomicAdd(&s_output[i][(z+x+4 )&63], input_smem[t][z]*filter[tx+256 +(k<<12)]);
            atomicAdd(&s_output[i][(z+x+8 )&63], input_smem[t][z]*filter[tx+512 +(k<<12)]);
            atomicAdd(&s_output[i][(z+x+12)&63], input_smem[t][z]*filter[tx+768 +(k<<12)]);
            atomicAdd(&s_output[i][(z+x+16)&63], input_smem[t][z]*filter[tx+1024+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+20)&63], input_smem[t][z]*filter[tx+1280+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+24)&63], input_smem[t][z]*filter[tx+1536+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+28)&63], input_smem[t][z]*filter[tx+1792+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+32)&63], input_smem[t][z]*filter[tx+2048+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+36)&63], input_smem[t][z]*filter[tx+2304+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+40)&63], input_smem[t][z]*filter[tx+2560+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+44)&63], input_smem[t][z]*filter[tx+2816+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+48)&63], input_smem[t][z]*filter[tx+3072+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+52)&63], input_smem[t][z]*filter[tx+3328+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+56)&63], input_smem[t][z]*filter[tx+3584+(k<<12)]);
            atomicAdd(&s_output[i][(z+x+60)&63], input_smem[t][z]*filter[tx+3840+(k<<12)]);
        }
    }
    
    __syncthreads();
    if(tx < 64) {
        const int out_start1 = ((bx<<1)+0) + (((by<<1)+0)<<3) + ((tx)<<6);
        const int out_start2 = ((bx<<1)+1) + (((by<<1)+0)<<3) + ((tx)<<6);
        const int out_start3 = ((bx<<1)+0) + (((by<<1)+1)<<3) + ((tx)<<6);
        const int out_start4 = ((bx<<1)+1) + (((by<<1)+1)<<3) + ((tx)<<6);
        output[out_start1] = clamp(((s_output[0 ][tx] + (1 << 4)) >>5)) + 128;
        output[out_start2] = clamp(((s_output[1 ][tx] + (1 << 4)) >>5)) + 128;
        output[out_start3] = clamp(((s_output[2 ][tx] + (1 << 4)) >>5)) + 128;
        output[out_start4] = clamp(((s_output[3 ][tx] + (1 << 4)) >>5)) + 128;
    }
}


__global__ void winograd( signed char *input,  signed short *weight,  signed char *output){
    const int tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;
    const int z = tx&63, x=tx>>6;
	__shared__ signed char input_smem [16][64];
	__shared__ int BtdB [16][64];
	__shared__ int I [16][64];

    const int start_xy= ((bx<<1)<<6) + ((by<<1)*640) + tx;
    input_smem[x   ][z] = input[start_xy];
    input_smem[x+4 ][z] = input[start_xy+640];
    input_smem[x+8 ][z] = input[start_xy+1280];
    input_smem[x+12][z] = input[start_xy+1920];
    I[x   ][z] = 0;
    I[x+4 ][z] = 0;
    I[x+8 ][z] = 0;
    I[x+12][z] = 0;
    
    __syncthreads();
    switch (x)
    {
    case 0:
        BtdB[0 ][z] =  input_smem[0 ][z] - input_smem[8 ][z] - input_smem[2 ][z] + input_smem[10][z];
        BtdB[1 ][z] =  input_smem[1 ][z] - input_smem[9 ][z] + input_smem[2 ][z] - input_smem[10][z];
        BtdB[2 ][z] = -input_smem[1 ][z] + input_smem[9 ][z] + input_smem[2 ][z] - input_smem[10][z];
        BtdB[3 ][z] =  input_smem[1 ][z] - input_smem[9 ][z] - input_smem[3 ][z] + input_smem[11][z];
        break;
    case 1:
        BtdB[4 ][z] =  input_smem[4 ][z] + input_smem[8 ][z] - input_smem[6 ][z] - input_smem[10][z];
        BtdB[5 ][z] =  input_smem[5 ][z] + input_smem[9 ][z] + input_smem[6 ][z] + input_smem[10][z];
        BtdB[6 ][z] = -input_smem[5 ][z] - input_smem[9 ][z] + input_smem[6 ][z] + input_smem[10][z];
        BtdB[7 ][z] =  input_smem[5 ][z] + input_smem[9 ][z] - input_smem[7 ][z] - input_smem[11][z];
        break;
    case 2:
        //(2,0), (2,1), (2,2), (2,3)
        BtdB[8 ][z] = -input_smem[4 ][z] + input_smem[8 ][z] + input_smem[6 ][z] - input_smem[10][z];
        BtdB[9 ][z] = -input_smem[5 ][z] + input_smem[9 ][z] - input_smem[6 ][z] + input_smem[10][z];
        BtdB[10][z] =  input_smem[5 ][z] - input_smem[9 ][z] - input_smem[6 ][z] + input_smem[10][z];
        BtdB[11][z] = -input_smem[5 ][z] + input_smem[9 ][z] + input_smem[7 ][z] - input_smem[11][z];
        break;
    case 3:
        //(3,0), (3,1), (3,2), (3,3)
        BtdB[12][z] =  input_smem[4 ][z] - input_smem[12][z] - input_smem[6 ][z] + input_smem[14][z];
        BtdB[13][z] =  input_smem[5 ][z] - input_smem[13][z] + input_smem[6 ][z] - input_smem[14][z];
        BtdB[14][z] = -input_smem[5 ][z] + input_smem[13][z] + input_smem[6 ][z] - input_smem[14][z];
        BtdB[15][z] =  input_smem[5 ][z] - input_smem[13][z] - input_smem[7 ][z] + input_smem[15][z];
        break;
    }
    
    __syncthreads();
    for(int i=0; i<16; i++){  //xyごとに計算する
        atomicAdd(&I[i][(z+x   )&63], BtdB[i][z]*weight[tx    +(i<<12)]);
        atomicAdd(&I[i][(z+x+4 )&63], BtdB[i][z]*weight[tx+256+(i<<12)]);
        atomicAdd(&I[i][(z+x+8 )&63], BtdB[i][z]*weight[tx+512+(i<<12)]);
        atomicAdd(&I[i][(z+x+12)&63], BtdB[i][z]*weight[tx+768+(i<<12)]);
        atomicAdd(&I[i][(z+x+16)&63], BtdB[i][z]*weight[tx+1024+(i<<12)]);
        atomicAdd(&I[i][(z+x+20)&63], BtdB[i][z]*weight[tx+1280+(i<<12)]);
        atomicAdd(&I[i][(z+x+24)&63], BtdB[i][z]*weight[tx+1536+(i<<12)]);
        atomicAdd(&I[i][(z+x+28)&63], BtdB[i][z]*weight[tx+1792+(i<<12)]);
        atomicAdd(&I[i][(z+x+32)&63], BtdB[i][z]*weight[tx+2048+(i<<12)]);
        atomicAdd(&I[i][(z+x+36)&63], BtdB[i][z]*weight[tx+2304+(i<<12)]);
        atomicAdd(&I[i][(z+x+40)&63], BtdB[i][z]*weight[tx+2560+(i<<12)]);
        atomicAdd(&I[i][(z+x+44)&63], BtdB[i][z]*weight[tx+2816+(i<<12)]);
        atomicAdd(&I[i][(z+x+48)&63], BtdB[i][z]*weight[tx+3072+(i<<12)]);
        atomicAdd(&I[i][(z+x+52)&63], BtdB[i][z]*weight[tx+3328+(i<<12)]);
        atomicAdd(&I[i][(z+x+56)&63], BtdB[i][z]*weight[tx+3584+(i<<12)]);
        atomicAdd(&I[i][(z+x+60)&63], BtdB[i][z]*weight[tx+3840+(i<<12)]);
    }

    __syncthreads();
    if(tx < 64) {
        const int out_start1 = ((bx<<1)+0) + (((by<<1)+0)<<3) + ((tx)<<6);
        const int out_start2 = ((bx<<1)+1) + (((by<<1)+0)<<3) + ((tx)<<6);
        const int out_start3 = ((bx<<1)+0) + (((by<<1)+1)<<3) + ((tx)<<6);
        const int out_start4 = ((bx<<1)+1) + (((by<<1)+1)<<3) + ((tx)<<6);
        output[out_start1] = clamp((((I[0 ][tx] + I[1 ][tx] + I[2 ][tx] + I[4 ][tx] + I[5 ][tx] + I[6 ][tx] + I[8 ][tx] + I[9 ][tx] + I[10][tx]) + (1 << 6)) >>7)) + 128;
        output[out_start2] = clamp((((I[1 ][tx] - I[2 ][tx] - I[3 ][tx] + I[5 ][tx] - I[6 ][tx] - I[7 ][tx] + I[9 ][tx] - I[10][tx] - I[11][tx]) + (1 << 6)) >>7)) + 128;
        output[out_start3] = clamp((((I[4 ][tx] + I[5 ][tx] + I[6 ][tx] - I[8 ][tx] - I[9 ][tx] - I[10][tx] - I[12][tx] - I[13][tx] - I[14][tx]) + (1 << 6)) >>7)) + 128;
        output[out_start4] = clamp((((I[5 ][tx] - I[6 ][tx] - I[7 ][tx] - I[9 ][tx] + I[10][tx] + I[11][tx] - I[13][tx] + I[14][tx] + I[15][tx]) + (1 << 6)) >>7)) + 128;
    }
}



int main(){
    cudaEvent_t start, stop;
    float elapsed_time_ms1, elapsed_time_ms2;
    signed char *h_char = ( signed char *)malloc(SIZE * sizeof( signed char));
    initialData(h_char, SIZE);

    // allocate global memory
    signed char *d_char, *d_char_out, *d_filter;
    signed short *d_wino;
    cudaMalloc( (void **) &d_char, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_char_out, SIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_filter, FSIZE * sizeof( signed char) );
    cudaMalloc( (void **) &d_wino, FSIZE_WINO * sizeof( signed short ) );

    cudaMemcpy( d_char, h_char, SIZE * sizeof( signed char), cudaMemcpyHostToDevice );

    signed char f;
    signed short f_short;
	FILE* fp;
    signed char x1_1[FSIZE];
    signed short wino[FSIZE_WINO];
    fp = fopen( "./params/layer3.0.conv2.weight", "rb" );
    if (!fp) printf("x1_1: pathを間違えています\n");
    for(int i=0; i<FSIZE; i++){
        if( fread( &f, sizeof(f), 1, fp ) < 1 ){
            fputs( "x1_1: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        x1_1[i] = f;
    }
    signed char temp[9][INCH*OUTCH];
    for (int i = 0; i < FSIZE; i++){
        int ch = (i/9)%INCH, xy = (i%9), n_th = i/(9*INCH);
        temp[xy][ch + n_th*INCH] = x1_1[i];
    }

    for(int i=0;i<9;i++){
        for(int k=0;k<INCH*OUTCH;k++){
            int n_th = ((k%OUTCH + (k/OUTCH)))%OUTCH, ch = k%INCH;
            x1_1[i*INCH*OUTCH + k] =  temp[i][n_th*INCH+ch];
        }
    }
    cudaMemcpy(d_filter, x1_1, sizeof( signed char) * FSIZE, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);


    fp = fopen( "./wino_params_short/layer3.0.conv2.weight", "rb" );
    if (!fp) printf("wino: pathを間違えています\n");
    for(int i=0; i<FSIZE_WINO; i++){
        if( fread( &f_short, sizeof(f_short), 1, fp ) < 1 ){
            fputs( "wino: 読み込み中にエラーが発生しました。\n", stderr );
            exit( EXIT_FAILURE );
        }
        wino[i] = f_short;
    }
    signed short t[16][INCH*OUTCH];
    for (int i = 0; i < FSIZE_WINO; i++){
        int ch = (i/16)%INCH, xy = (i%16), n_th = i/(16*INCH);
        t[xy][ch + n_th*INCH] = wino[i];
    }

    for(int i=0;i<16;i++){
        for(int k=0;k<INCH*OUTCH;k++){
            int n_th = ((k%OUTCH + (k/OUTCH)))%OUTCH, ch = k%INCH;
            wino[i*INCH*OUTCH + k] =  t[i][n_th*INCH+ch];
        }
    }
    cudaMemcpy(d_wino, wino, sizeof(signed short) * FSIZE_WINO, cudaMemcpyHostToDevice);
    if (fp) fclose(fp);
    

    //warm up
    for(int i=0;i<100;i++) conv<<<dim3(4, 4), 256>>>(d_char, d_filter, d_char_out);
    //Measure
    cudaMemset(&d_char_out, 0, sizeof(signed char)*SIZE);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    conv<<<dim3(4, 4), 256>>>(d_char, d_filter, d_char_out);
    elapsed_time_ms1=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms1, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("normal:%f\n", elapsed_time_ms1);

    signed char res[SIZE];
    cudaMemcpy(res, d_char_out, sizeof(signed char) * SIZE, cudaMemcpyDeviceToHost);
    

    //Measure
    cudaMemset(&d_char_out, 0, sizeof(signed char)*(SIZE));
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    winograd<<<dim3(4, 4), 256>>>(d_char, d_wino, d_char_out);
    elapsed_time_ms2=0.0f;
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms2, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("winograd:%f\n", elapsed_time_ms2);
    
    signed char res1[SIZE];
    cudaMemcpy(res1, d_char_out, sizeof(signed char) * SIZE, cudaMemcpyDeviceToHost);


    //check result
    int miss = 0;
    // for(int i=0;i<SIZE; i++) if(res[i] != res1[i]) {printf("%d ", i); miss++;}
    for(int i=0;i<SIZE; i++) if(res[i] != res1[i]) {miss++;}
    if(miss == 0) printf("%f 倍速くなりました。", elapsed_time_ms1/elapsed_time_ms2);
    else if(miss != 0) printf("%f 倍速くなりました。答え一致してないけどね\n miss = %d", elapsed_time_ms1/elapsed_time_ms2, miss);
    
    free(h_char);
    cudaFree(d_char);
    cudaFree(d_char_out);
    cudaFree(d_filter);
    cudaFree(d_wino);

    return 0;
    
}