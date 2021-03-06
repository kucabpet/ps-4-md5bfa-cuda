#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>

#define COUNT_UNIT8_T_HASH 16

#define MD5_CHUNKS_BYTE 512/8
#define MD5_TEXT_LEN 448/8

//#define DEBUG

#define LEFTROTATE(x, c) (((x) << (c)) | ((x) >> (32 - (c))))

__host__ __device__
void to_bytes(uint32_t val, uint8_t *bytes) {
    bytes[0] = (uint8_t) val;
    bytes[1] = (uint8_t) (val >> 8);
    bytes[2] = (uint8_t) (val >> 16);
    bytes[3] = (uint8_t) (val >> 24);
}

__host__ __device__
uint32_t to_int32(const uint8_t *bytes) {
    return (uint32_t) bytes[0] | ((uint32_t) bytes[1] << 8)
           | ((uint32_t) bytes[2] << 16) | ((uint32_t) bytes[3] << 24);
}

__host__ __device__
void md5(const uint8_t *initial_msg, size_t initial_len, uint8_t *digest) {

    // Constants are the integer part of the sines of integers (in radians) * 2^32.
    const uint32_t k[64] = {0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
                            0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af,
                            0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
                            0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453,
                            0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
                            0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681,
                            0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
                            0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5,
                            0x1fa27cf8, 0xc4ac5665, 0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
                            0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0,
                            0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};

    // r specifies the per-round shift amounts
    const uint32_t r[] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17,
                          22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16,
                          23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15, 21, 6, 10,
                          15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

    // These vars will contain the hash
    uint32_t h0, h1, h2, h3;

    // Message (to prepare)
    uint8_t *msg = NULL;

    size_t new_len, offset;
    uint32_t w[16];
    uint32_t a, b, c, d, i, f, g, temp;

    // Initialize variables - simple count in nibbles:
    h0 = 0x67452301;
    h1 = 0xefcdab89;
    h2 = 0x98badcfe;
    h3 = 0x10325476;

    //Pre-processing:
    //append "1" bit to message
    //append "0" bits until message length in bits ≡ 448 (mod 512)
    //append length mod (2^64) to message

    for (new_len = initial_len + 1; new_len % (MD5_CHUNKS_BYTE) != MD5_TEXT_LEN; new_len++);

    msg = (uint8_t *) malloc(new_len + 8);
    memcpy(msg, initial_msg, initial_len);
    msg[initial_len] = 0x80; // append the "1" bit; most significant bit is "first"

    for (offset = initial_len + 1; offset < new_len; offset++)
        msg[offset] = 0; // append "0" bits

    // append the len in bits at the end of the buffer.
    to_bytes(initial_len * 8, msg + new_len);

    // initial_len>>29 == initial_len*8>>32, but avoids overflow.
    to_bytes(initial_len >> 29, msg + new_len + 4);

    // Process the message in successive 512-bit chunks:
    //for each 512-bit chunk of message:
    for (offset = 0; offset < new_len; offset += (MD5_CHUNKS_BYTE)) {

        // break chunk into sixteen 32-bit words w[j], 0 ≤ j ≤ 15
        for (i = 0; i < COUNT_UNIT8_T_HASH; i++)
            w[i] = to_int32(msg + offset + i * 4);

        // Initialize hash value for this chunk:
        a = h0;
        b = h1;
        c = h2;
        d = h3;

        // Main loop:
        for (i = 0; i < 64; i++) {

            if (i < 16) {
                f = (b & c) | ((~b) & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | ((~d) & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7 * i) % 16;
            }

            temp = d;
            d = c;
            c = b;
            b = b + LEFTROTATE((a + f + k[i] + w[g]), r[i]);
            a = temp;

        }

        // Add this chunk's hash to result so far:
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;

    }

    // cleanup
    free(msg);

    //var char digest[16] := h0 append h1 append h2 append h3 //(Output is in little-endian)
    to_bytes(h0, digest);
    to_bytes(h1, digest + 4);
    to_bytes(h2, digest + 8);
    to_bytes(h3, digest + 12);
}

__host__ __device__
int my_strlen(char *text) {
    int len = 0;
    int i = 0;
    while (text[i++] != '\0') {
        len++;
    }
    return len;
}

__host__ __device__
void hash_md5(char *input, uint8_t *result) {
    md5((uint8_t *) input, (size_t) my_strlen(input), result);
}


__global__ void kernel_mult(char *words, const int height, const int width, uint8_t *cHashWords) {

    int l = blockDim.x * blockIdx.x + threadIdx.x;

    if (l >= height) {
        return;
    }

    char *word = new char[width + 1];

    int j;
    for (j = 0; j < width; j++) {
        word[j] = words[width * l + j];
    }
    word[j] = '\0';

    hash_md5(word, &cHashWords[COUNT_UNIT8_T_HASH * l]);

#ifdef DEBUG
    printf("debug: hashing word: %s\n", word);
#endif

    delete[] word;
}


void run_mult(char *words, const int height, const int width, uint8_t *hashed_words) {

    cudaError_t cerr;
    int threads = 128;
    int length = height * width;
    int blocks = (length + threads - 1) / threads;

    char *cWords;
    cerr = cudaMalloc(&cWords, length * sizeof(char));
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

    uint8_t *cHashWords;
    cerr = cudaMalloc(&cHashWords, height * COUNT_UNIT8_T_HASH * sizeof(uint8_t));
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));


    cerr = cudaMemcpy(cWords, words, length * sizeof(char), cudaMemcpyHostToDevice);
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

    kernel_mult<<< blocks, threads >>> (cWords, height, width, cHashWords);

    if ((cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

    cerr = cudaMemcpy(hashed_words, cHashWords, height * COUNT_UNIT8_T_HASH * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(cerr));

    cudaFree(cWords);
    cudaFree(cHashWords);
}
