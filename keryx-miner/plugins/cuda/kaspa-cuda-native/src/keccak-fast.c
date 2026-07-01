/* keccak-f[1600] rapido para GPU: estado residente en registros (indices
 * compile-time), todo desenrollado, rho/pi explicito sin lookups de byte ni
 * cadena serial sobre arrays en local memory. Interfaz hash() identica a
 * keccak-tiny.c (drop-in) para el kernel kHeavyHash de Keryx. Bit-exacto. */
#include <stdint.h>
#include <string.h>

#define Plen 200

__device__ __constant__ static const uint64_t RC_F[24] = {
  1ULL, 0x8082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
  0x808bULL, 0x80000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
  0x8aULL, 0x88ULL, 0x80008009ULL, 0x8000000aULL,
  0x8000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
  0x8000000000008002ULL, 0x8000000000000080ULL, 0x800aULL, 0x800000008000000aULL,
  0x8000000080008081ULL, 0x8000000000008080ULL, 0x80000001ULL, 0x8000000080008008ULL};

#define ROTL64(x,y) (((x) << (y)) | ((x) >> (64 - (y))))

__device__ __forceinline__ static void keccakf(void* state) {
  uint64_t* st = (uint64_t*)state;
  uint64_t bc0,bc1,bc2,bc3,bc4,t0,t1,t2,t3,t4,t;
  #pragma unroll
  for (int r = 0; r < 24; r++) {
    /* Theta */
    bc0 = st[0]^st[5]^st[10]^st[15]^st[20];
    bc1 = st[1]^st[6]^st[11]^st[16]^st[21];
    bc2 = st[2]^st[7]^st[12]^st[17]^st[22];
    bc3 = st[3]^st[8]^st[13]^st[18]^st[23];
    bc4 = st[4]^st[9]^st[14]^st[19]^st[24];
    t0 = bc4 ^ ROTL64(bc1,1); t1 = bc0 ^ ROTL64(bc2,1);
    t2 = bc1 ^ ROTL64(bc3,1); t3 = bc2 ^ ROTL64(bc4,1); t4 = bc3 ^ ROTL64(bc0,1);
    st[0]^=t0; st[5]^=t0; st[10]^=t0; st[15]^=t0; st[20]^=t0;
    st[1]^=t1; st[6]^=t1; st[11]^=t1; st[16]^=t1; st[21]^=t1;
    st[2]^=t2; st[7]^=t2; st[12]^=t2; st[17]^=t2; st[22]^=t2;
    st[3]^=t3; st[8]^=t3; st[13]^=t3; st[18]^=t3; st[23]^=t3;
    st[4]^=t4; st[9]^=t4; st[14]^=t4; st[19]^=t4; st[24]^=t4;
    /* Rho + Pi (permutacion explicita, constantes en immediates) */
    t = st[1];
    st[ 1]=ROTL64(st[ 6],44); st[ 6]=ROTL64(st[ 9],20); st[ 9]=ROTL64(st[22],61);
    st[22]=ROTL64(st[14],39); st[14]=ROTL64(st[20],18); st[20]=ROTL64(st[ 2],62);
    st[ 2]=ROTL64(st[12],43); st[12]=ROTL64(st[13],25); st[13]=ROTL64(st[19], 8);
    st[19]=ROTL64(st[23],56); st[23]=ROTL64(st[15],41); st[15]=ROTL64(st[ 4],27);
    st[ 4]=ROTL64(st[24],14); st[24]=ROTL64(st[21], 2); st[21]=ROTL64(st[ 8],55);
    st[ 8]=ROTL64(st[16],45); st[16]=ROTL64(st[ 5],36); st[ 5]=ROTL64(st[ 3],28);
    st[ 3]=ROTL64(st[18],21); st[18]=ROTL64(st[17],15); st[17]=ROTL64(st[11],10);
    st[11]=ROTL64(st[ 7], 6); st[ 7]=ROTL64(st[10], 3); st[10]=ROTL64(t,      1);
    /* Chi */
    bc0=st[0];bc1=st[1];bc2=st[2];bc3=st[3];bc4=st[4];
    st[0]=bc0^((~bc1)&bc2); st[1]=bc1^((~bc2)&bc3); st[2]=bc2^((~bc3)&bc4); st[3]=bc3^((~bc4)&bc0); st[4]=bc4^((~bc0)&bc1);
    bc0=st[5];bc1=st[6];bc2=st[7];bc3=st[8];bc4=st[9];
    st[5]=bc0^((~bc1)&bc2); st[6]=bc1^((~bc2)&bc3); st[7]=bc2^((~bc3)&bc4); st[8]=bc3^((~bc4)&bc0); st[9]=bc4^((~bc0)&bc1);
    bc0=st[10];bc1=st[11];bc2=st[12];bc3=st[13];bc4=st[14];
    st[10]=bc0^((~bc1)&bc2); st[11]=bc1^((~bc2)&bc3); st[12]=bc2^((~bc3)&bc4); st[13]=bc3^((~bc4)&bc0); st[14]=bc4^((~bc0)&bc1);
    bc0=st[15];bc1=st[16];bc2=st[17];bc3=st[18];bc4=st[19];
    st[15]=bc0^((~bc1)&bc2); st[16]=bc1^((~bc2)&bc3); st[17]=bc2^((~bc3)&bc4); st[18]=bc3^((~bc4)&bc0); st[19]=bc4^((~bc0)&bc1);
    bc0=st[20];bc1=st[21];bc2=st[22];bc3=st[23];bc4=st[24];
    st[20]=bc0^((~bc1)&bc2); st[21]=bc1^((~bc2)&bc3); st[22]=bc2^((~bc3)&bc4); st[23]=bc3^((~bc4)&bc0); st[24]=bc4^((~bc0)&bc1);
    /* Iota */
    st[0]^=RC_F[r];
  }
}

#define P keccakf

/* misma firma fija que keccak-tiny.c: absorbe 1 bloque (initP ^ in[0..9]),
 * permuta, exprime 32 bytes. */
__device__ __forceinline__ static void hash(const uint8_t initP[Plen],
                                             uint8_t* out, const uint8_t* in) {
  uint64_t a[25];
  #pragma unroll
  for (int i=0;i<10;i++) a[i] = ((const uint64_t*)initP)[i] ^ ((const uint64_t*)in)[i];
  #pragma unroll
  for (int i=10;i<25;i++) a[i] = ((const uint64_t*)initP)[i];
  P(a);
  #pragma unroll
  for (int i=0;i<4;i++) ((uint64_t*)out)[i] = a[i];
}
