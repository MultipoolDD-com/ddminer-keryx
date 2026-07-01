// Benchmark + verificacion del kernel heavy_hash de Keryx (kHeavyHash + wave_mix).
// Mide MH/s aislando el kernel, e imprime el hash de un nonce fijo para verificar
// bit-exactitud entre la version stock y la optimizada.
#include "kaspa-cuda.cu"
#include <cstdio>
#include <cstring>

// kernel auxiliar: en vez de comparar con target, escribe el hash final de cada
// nonce a un buffer (para verificar). Reusa toda la cadena del kernel real.
__global__ void hash_dump(uint64_t base_nonce, uint64_t n, uint256_t* out) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= n) return;
    uint64_t nonce = base_nonce + id;
    uint8_t input[80];
    memcpy(input, hash_header, HASH_HEADER_SIZE);
    uint256_t hash_;
    memcpy(input + HASH_HEADER_SIZE, (uint8_t*)(&nonce), 8);
    hash(powP, hash_.hash, input);
    uchar4 packed_hash[QUARTER_MATRIX_SIZE] = {0};
    #pragma unroll
    for (int i = 0; i < QUARTER_MATRIX_SIZE; i++)
        packed_hash[i] = make_uchar4((hash_.hash[2*i]&0xF0)>>4,(hash_.hash[2*i]&0x0F),
                                     (hash_.hash[2*i+1]&0xF0)>>4,(hash_.hash[2*i+1]&0x0F));
    uint32_t p1, p2;
    #pragma unroll
    for (int rowId = 0; rowId < HALF_MATRIX_SIZE; rowId++) {
        amul4bit((uint32_t*)(matrix[2*rowId]), (uint32_t*)packed_hash, &p1);
        amul4bit((uint32_t*)(matrix[2*rowId+1]), (uint32_t*)packed_hash, &p2);
        p1 >>= 6; p1 &= 0xF0; p2 >>= 10;
        hash_.hash[rowId] ^= ((uint8_t)p1 | (uint8_t)p2);
    }
    wave_mix(&hash_);
    uint8_t in2[80]; memset(in2,0,80); memcpy(in2, hash_.hash, 32);
    hash(heavyP, hash_.hash, in2);
    out[id] = hash_;
}

int main(int argc, char** argv) {
    // datos arbitrarios pero deterministas
    uint8_t h_header[72]; for (int i=0;i<72;i++) h_header[i]=(uint8_t)(i*7+1);
    uint8_t h_matrix[64][64]; for(int i=0;i<64;i++)for(int j=0;j<64;j++) h_matrix[i][j]=(uint8_t)((i*13+j*7)&0x0F);
    uint256_t h_target; for(int i=0;i<4;i++) h_target.number[i]=0xFFFFFFFFFFFFFFFFULL; // nunca golpea -> corre completo

    cudaMemcpyToSymbol(hash_header, h_header, 72);
    cudaMemcpyToSymbol(matrix, h_matrix, 64*64);
    cudaMemcpyToSymbol(target, &h_target, sizeof(h_target));

    // ---- modo verificacion: imprime hash de 4 nonces fijos ----
    if (argc > 1 && strcmp(argv[1], "verify") == 0) {
        int N = 8;
        uint256_t* d_out; cudaMalloc(&d_out, N*sizeof(uint256_t));
        hash_dump<<<1, N>>>(1000ULL, N, d_out);
        cudaDeviceSynchronize();
        uint256_t h_out[8]; cudaMemcpy(h_out, d_out, N*sizeof(uint256_t), cudaMemcpyDeviceToHost);
        for (int n=0;n<N;n++){ printf("nonce %d: ", 1000+n); for(int i=0;i<32;i++) printf("%02x", h_out[n].hash[i]); printf("\n"); }
        return 0;
    }

    // ---- modo benchmark ----
    #ifndef BLK
#define BLK 512
#endif
#ifndef GM
#define GM 16
#endif
    int block=BLK, grid=0; cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    grid = p.multiProcessorCount * GM;          // muchos bloques residentes
    uint64_t threads = (uint64_t)block*grid;
    uint64_t* d_states; cudaMalloc(&d_states,8); uint64_t s=0x1234; cudaMemcpy(d_states,&s,8,cudaMemcpyHostToDevice);
    uint64_t* d_nonce; cudaMalloc(&d_nonce,8);
    // warmup
    heavy_hash<<<grid,block>>>(~0ULL, 0ULL, threads, RANDOM_LEAN, d_states, d_nonce);
    cudaDeviceSynchronize();
    if (cudaGetLastError()){ printf("ERR launch\n"); return 1; }

    int iters = 200;
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    for (int it=0; it<iters; it++)
        heavy_hash<<<grid,block>>>(~0ULL, 0ULL, threads, RANDOM_LEAN, d_states, d_nonce);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms=0; cudaEventElapsedTime(&ms,e0,e1);
    double total = (double)threads*iters;
    double mhs = total/(ms/1000.0)/1e6;
    printf("GPU: %s  SMs=%d\n", p.name, p.multiProcessorCount);
    printf("threads/launch=%llu  iters=%d  tiempo=%.1fms\n",(unsigned long long)threads,iters,ms);
    printf(">>> HASHRATE: %.1f MH/s\n", mhs);
    return 0;
}
