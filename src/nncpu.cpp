#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>

#if defined(USE_AVX2)
#include <immintrin.h>
#endif

#define DLL_EXPORT
#include "nncpu.h"
#undef DLL_EXPORT

#ifdef _MSC_VER
#  define _CRT_SECURE_NO_WARNINGS
#  pragma warning (disable: 4996)
#endif
/*
types and quantization
*/
#if 1

#define SCALE_WEIGHT  64
#define SCALE_ACT     127
#define SCALE_BIAS    (SCALE_WEIGHT*SCALE_ACT)

typedef int8_t  weight_t;
typedef int32_t bias_t;
typedef int8_t  output_t;
typedef int16_t input_weight_t;

#define QUANTIZE       1

#else

#define SCALE_WEIGHT  1
#define SCALE_ACT     1
#define SCALE_BIAS    (SCALE_WEIGHT*SCALE_ACT)

typedef float weight_t;
typedef float bias_t;
typedef float output_t;
typedef float input_weight_t;

#define QUANTIZE       0
#endif

/*
Force inline
*/
#if defined (__GNUC__)
#   define INLINE  __inline __attribute__((always_inline))
#elif defined (_WIN32)
#   define INLINE  __forceinline
#else
#   define INLINE  __inline
#endif

/*
weights and biases
*/
#define CACHE_ALIGN alignas(64)

CACHE_ALIGN static input_weight_t input_weights[32*12*64*256]; //order: [N_inp][N_out]
CACHE_ALIGN static input_weight_t input_biases[256];

CACHE_ALIGN static weight_t hidden1_weights[32*512];           //order: [N_out][N_inp]
CACHE_ALIGN static bias_t hidden1_biases[32];

CACHE_ALIGN static weight_t hidden2_weights[32*32];            //order: [N_out][N_inp]
CACHE_ALIGN static bias_t hidden2_biases[32];

CACHE_ALIGN static weight_t output_weights[1*32];              //order: [N_out][N_inp]
CACHE_ALIGN static bias_t output_biases[1];


/*Basic vector operations that vectorize well
  without simd intrinsics
  */
template<int n_input, typename T>
INLINE void addVectors( 
    T* __restrict const p1,
    const T* __restrict const p2)
{
    for(int i = 0; i < n_input; i++)
        p1[i] += p2[i];
}

template<int n_input, typename T>
INLINE void clampVector( T* const  p)
{
#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))

    for(int i = 0; i < n_input; i++)
        p[i] = clamp(p[i],0,SCALE_ACT);

#undef clamp
}

template<int n_input, typename T>
INLINE void sigmoidVector( T* const  p)
{
    for(int i = 0; i < n_input; i++)
        p[i] = 1 / (1 + exp(-float(p[i]) / SCALE_ACT));
}

/*Compute dot product of two vectors.
  This may not parallelize well */
#if !QUANTIZE

template<int offsetRegs>
INLINE __m256 fma32( __m256 acc, const float* p1, const float* p2 )
{
    constexpr int lanes = offsetRegs * 8;
    const __m256 a = _mm256_load_ps( p1 + lanes );
    const __m256 b = _mm256_load_ps( p2 + lanes );
    return _mm256_fmadd_ps( a, b, acc );
}

template<int n_input>
float dotProduct( const float* p1, const float* p2)
{
#if !defined(USE_AVX2)
    float sum = 0;
    for(int i = 0; i < n_input; i++)
        sum += p1[i] * p2[i];
    return sum;
#else
    __m256 dot0 = _mm256_setzero_ps();
    __m256 dot1 = dot0;
    __m256 dot2 = dot0;
    __m256 dot3 = dot0;

    for(int i = 0; i < n_input; i += 32)
    {
        dot0 = fma32<0>( dot0, p1, p2 );
        dot1 = fma32<1>( dot1, p1, p2 );
        dot2 = fma32<2>( dot2, p1, p2 );
        dot3 = fma32<3>( dot3, p1, p2 );
        p1 += 32;
        p2 += 32;
    }

    const __m256 dot01 = _mm256_add_ps( dot0, dot1 );
    const __m256 dot23 = _mm256_add_ps( dot2, dot3 );
    const __m256 dot0123 = _mm256_add_ps( dot01, dot23 );

    const __m128 r4 = _mm_add_ps( 
        _mm256_castps256_ps128( dot0123 ), _mm256_extractf128_ps( dot0123, 1 ) );
    const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
    const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
    return _mm_cvtss_f32( r1 );
#endif
}

#else

template<int offsetRegs>
INLINE __m256i fma8( const int8_t* p1, const int8_t* p2 )
{
    constexpr int lanes = offsetRegs * 32;
    __m256i *a = (__m256i *)(p1 + lanes);
    __m256i *b = (__m256i *)(p2 + lanes);
    __m256i prod = _mm256_maddubs_epi16(a[0], b[0]);
    prod = _mm256_madd_epi16(prod, _mm256_set1_epi16(1));
    return prod;
}

template<int n_input>
int32_t dotProduct( const int8_t* p1, const int8_t* p2)
{
#if !defined(USE_AVX2)
    int32_t sum = 0;
    for(int i = 0; i < n_input; i++)
        sum += p1[i] * p2[i];
    return sum;
#else
    __m256i prod = _mm256_setzero_si256();
    for(int i = 0; i < n_input; i += 32)
    {
        prod = _mm256_add_epi32(prod, fma8<0>( p1, p2 ));
        p1 += 32;
        p2 += 32;
    }
    __m128i sum = _mm_add_epi32(
        _mm256_castsi256_si128(prod), _mm256_extracti128_si256(prod, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x1b));
    return _mm_cvtsi128_si32(sum) + _mm_extract_epi32(sum, 1);
#endif
}

#endif

/*
Dense layer
*/
template<int n_input, int n_output, typename T>
void DENSE(
    const output_t* __restrict const input,
    const weight_t* __restrict const weights,
    const bias_t* __restrict const biases,
    T* __restrict const output
    )
{
    for(unsigned i = 0; i < n_output; i++) {
        bias_t sum = biases[i];
        sum += dotProduct<n_input>(input,weights + i * n_input);
        output[i] = sum / SCALE_WEIGHT;
    }
}

/*
compute top heads for each player
*/
#define MIRRORF64(sq)    ((sq) ^ 007)
#define MIRRORR64(sq)    ((sq) ^ 070)
#define file64(x)        ((x) &  7)
#define rank64(x)        ((x) >> 3)
#define INVERT(x)        (((x) > 6) ? ((x) - 6) : ((x) + 6))

INLINE void accumulate_input(
    const int player,
    const int* __restrict const pieces,
    const int* __restrict const squares,
    input_weight_t* __restrict const accumulator,
    output_t* __restrict const output
    ) {
    
    //compute king index (0 to 31)
    const bool flip_rank = (player == 1);
    const unsigned ksq = squares[player];
    int f = file64(ksq);
    int r = rank64(ksq);
    const bool flip_file = (f < 4);
    if(flip_rank) r = 7 - r;
    if(flip_file) f = 7 - f;
    const unsigned kidx = (r * 4 + (f - 4));

    //initialize accumulater
    memcpy(accumulator, input_biases, sizeof(input_biases));

    //add contributions of each piece
    for(unsigned i = 0, pc; (pc = pieces[i]) != 0; i++) {
        unsigned sq = squares[i];
        if(flip_rank) {
            sq = MIRRORR64(sq);
            pc = INVERT(pc);
        }
        if(flip_file) {
            sq = MIRRORF64(sq);
        }

        input_weight_t* const pinp = input_weights + 
                        kidx*12*64*256 + (pc-1)*64*256 + sq*256;
        addVectors<256,input_weight_t>(accumulator,pinp);
    }

    //activation
    for(unsigned i = 0; i < 256; i++)
        output[i] = accumulator[i] / SCALE_WEIGHT;

    clampVector<256,output_t>(output);
}

/*
evaluate net
*/
static INLINE float logit(float p) {
    if(p < 1e-15) p = 1e-15;
    else if(p > 1 - 1e-15) p = 1 - 1e-15;
    return log((1 - p) / p) / (-0.00575646273);
}

DLLExport int _CDECL nncpu_evaluate(int player, int* pieces, int* squares)
{
    //output_t
    CACHE_ALIGN input_weight_t accumulator[2*256];
    CACHE_ALIGN output_t input_output[2*256];
    CACHE_ALIGN output_t hidden1_output[32];
    CACHE_ALIGN output_t hidden2_output[32];
    float score[1];

    //accumulate player and opponent heads
    accumulate_input(player,pieces,squares,accumulator,input_output);
    accumulate_input(1-player,pieces,squares,accumulator+256,input_output+256);

    //three dense layers
    DENSE<512,32,output_t>(input_output,hidden1_weights,hidden1_biases,hidden1_output);
    clampVector<32,output_t>(hidden1_output);
    DENSE<32,32,output_t>(hidden1_output,hidden2_weights,hidden2_biases,hidden2_output);
    clampVector<32,output_t>(hidden2_output);
    DENSE<32,1,float>(hidden2_output,output_weights,output_biases,score);
    sigmoidVector<1,float>(score);

    return logit(score[0]);
}

/*
Read bytes in little endian byte order
*/
static float read_bytes(int count,FILE* f) {
    uint32_t x = 0;
    uint8_t* c = (uint8_t*) &x;
    for(int i = 0; i < count; i++)
        c[i] = ((uint8_t) fgetc(f));
    float* p = (float*) &x;
    return *p;
}
/*
Read weights.
   shuffle order of weights for best cache performance
*/
static void read_network(FILE* f)
{
    //version number
    read_bytes(sizeof(int),f);

    //input layer
    for(int sq = 0; sq < 64; sq++) {
        for(int kidx = 0; kidx < 32; kidx++) {
            for(int pc = 0; pc < 12; pc++) {
                for(int o = 0; o < 256; o++) {
                    float value = read_bytes(sizeof(float), f) * SCALE_BIAS;
                    input_weights[kidx*12*64*256 + pc*64*256 + sq*256+o] =
                        (input_weight_t)value;
                }
            }
        }
    }
    for(int o = 0; o < 256; o++) {
        float value = read_bytes(sizeof(float), f) * SCALE_BIAS;
        input_biases[o] = (input_weight_t)value;
    }

    //first hidden layer
    for(int i = 0; i < 512; i++) {
        for(int j = 0; j < 32; j++) {
            float value = read_bytes(sizeof(float), f) * SCALE_WEIGHT;
            hidden1_weights[j*512 + i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 32; i++) {
        float value = read_bytes(sizeof(float), f) * SCALE_BIAS;
        hidden1_biases[i] = (bias_t)value;
    }

    //second hidden layer
    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 32; j++){
            float value = read_bytes(sizeof(float), f) * SCALE_WEIGHT;
            hidden2_weights[j*32+i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 32; i++) {
        float value = read_bytes(sizeof(float), f) * SCALE_BIAS;
        hidden2_biases[i] = (bias_t)value;
    }

    //output layer
    for(int i = 0; i < 32; i++) {
        for(int j =0; j < 1; j++) {
            float value = read_bytes(sizeof(float), f) * SCALE_WEIGHT;
            output_weights[j*32+i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 1; i++) {
        float value = read_bytes(sizeof(float), f) * SCALE_BIAS;
        output_biases[i] = (bias_t)value;
    }
}

/*
init net
*/
DLLExport void _CDECL nncpu_init(const char* path)
{
    FILE* f = fopen(path, "rb");
    if(!f) {
        printf("NNCPU file not found!\n");
        fflush(stdout);
        return;
    } else {
        printf("Loading NNCPU : %s\n", path);
        fflush(stdout);
    }

    read_network(f);
    fclose(f);

    printf("NNCPU loaded !\n");
    fflush(stdout);
}
/*
Incremental evaluation @todo
*/
DLLExport int _CDECL nncpu_evaluate_incremental(
  int player, int* pieces, int* squares, NNCPUdata** nncpu)
{
  assert(nncpu[0] && uint64_t(&nncpu[0]->accumulator) % 64 == 0);

  return nncpu_evaluate(player,pieces,squares);
}
/*
Evaluate FEN
*/
static const char piece_name[] = "_KQRBNPkqrbnp_";
static const char rank_name[] = "12345678";
static const char file_name[] = "abcdefgh";
static const char col_name[] = "WwBb";
static const char cas_name[] = "KQkq";

void decode_fen(const char* fen_str, int* player, int* castle,
       int* fifty, int* move_number, int* piece, int* square)
{
  /*decode fen*/
  int sq,index = 2;
  const char* p = fen_str,*pfen;
  for(int r = 7;r >= 0; r--) {
      for(int f = 0;f <= 7;f++) {
          sq = r * 8 + f;
          if((pfen = strchr(piece_name,*p)) != 0) {
              int pc = int(strchr(piece_name,*pfen) - piece_name);
              if(pc == 1) {
                 piece[0] = pc;
                 square[0] = sq;
              } else if(pc == 7) {
                 piece[1] = pc;
                 square[1] = sq;
              } else {
                 piece[index] = pc;
                 square[index] = sq;
                 index++;
              }
          } else if((pfen = strchr(rank_name,*p)) != 0) {
              for(int i = 0;i < pfen - rank_name;i++) {
                  f++;
              }
          } 
          p++;
      }
      p++;
  }
  piece[index] = 0;
  square[index] = 0;

  /*player*/
  if((pfen = strchr(col_name,*p)) != 0)
      *player = ((pfen - col_name) >= 2);
  p++;
  p++;

  /*castling rights*/
  *castle = 0;
  if(*p == '-') {
      p++;
  } else {
      while((pfen = strchr(cas_name,*p)) != 0) {
          *castle |= (1 << (pfen - cas_name));
          p++;
      }
  }
  /*epsquare*/
  int epsquare;
  p++;
  if(*p == '-') {
      epsquare = 0;
      p++;
  } else {
      epsquare = int(strchr(file_name,*p) - file_name);
      p++;
      epsquare += 16 * int(strchr(rank_name,*p) - rank_name);
      p++;
  }
  square[index] = epsquare;

  /*fifty & hply*/
  p++;
  if(*p && *(p+1) && isdigit(*p) && ( isdigit(*(p+1)) || *(p+1) == ' ' ) ) {
      sscanf(p,"%d %d",fifty,move_number);
      if(*move_number <= 0) *move_number = 1;
  } else {
      *fifty = 0;
      *move_number = 1;
  }
}

DLLExport int _CDECL nncpu_evaluate_fen(const char* fen)
{
  int pieces[33],squares[33],player,castle,fifty,move_number;
  decode_fen((char*)fen,&player,&castle,&fifty,&move_number,pieces,squares);;
  return nncpu_evaluate(player,pieces,squares);
}