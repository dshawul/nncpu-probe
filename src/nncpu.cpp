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
#if !defined(USE_FLOAT)

#define SCALE_WEIGHT  64
#define SCALE_ACT     127
#define SCALE_BIAS    (SCALE_WEIGHT*SCALE_ACT)

typedef int8_t  weight_t;
typedef int32_t bias_t;
typedef int8_t  output_t;
typedef int16_t input_weight_t;

#else

#define SCALE_WEIGHT  1
#define SCALE_ACT     1
#define SCALE_BIAS    (SCALE_WEIGHT*SCALE_ACT)

typedef float weight_t;
typedef float bias_t;
typedef float output_t;
typedef float input_weight_t;

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
INLINE void subVectors( 
    T* __restrict const p1,
    const T* __restrict const p2)
{
    for(int i = 0; i < n_input; i++)
        p1[i] -= p2[i];
}

template<int n_input, typename IT, typename T>
INLINE void clampVector(
  const IT* __restrict const ip,
  T* __restrict const  p)
{
#define clamp(a, b, c) ((a) < (b) ? (b) : (a) > (c) ? (c) : (a))

    for(int i = 0; i < n_input; i++)
        p[i] = clamp(ip[i] / SCALE_WEIGHT,0,SCALE_ACT);

#undef clamp
}

/*Compute dot product of two vectors.
  This may not parallelize well */
#if defined(USE_FLOAT)

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
template<int n_input, int n_output>
void DENSE(
    const output_t* __restrict const input,
    const weight_t* __restrict const weights,
    const bias_t* __restrict const biases,
    bias_t* __restrict const output
    )
{
    memcpy(output, biases, n_output * sizeof(bias_t));
    for(unsigned i = 0; i < n_output; i++)
        output[i] += dotProduct<n_input>(input,weights + i * n_input);
}

/*
compute top heads for each player
*/
#define MIRRORF64(sq)    ((sq) ^ 007)
#define MIRRORR64(sq)    ((sq) ^ 070)
#define file64(x)        ((x) &  7)
#define rank64(x)        ((x) >> 3)
#define INVERT(x)        (((x) > 6) ? ((x) - 6) : ((x) + 6))
#define KING(c)          ((c) ? bking : wking)

//add/subtract influence of input node
#define INFLUENCE(op) {                                           \
  if(flip_rank) {                                                 \
      sq = MIRRORR64(sq);                                         \
      pc = INVERT(pc);                                            \
  }                                                               \
  if(flip_file) {                                                 \
      sq = MIRRORF64(sq);                                         \
  }                                                               \
  input_weight_t* const pinp = input_weights +                    \
                  kidx*12*64*256 + (pc-1)*64*256 + sq*256;        \
  op##Vectors<256,input_weight_t>(accumulation,pinp);             \
}

//compute king index (0 to 31)
#define KINDEX()                                                  \
    const bool flip_rank = (side == 1);                           \
    const unsigned ksq = pos->squares[side];                      \
    int f = file64(ksq);                                          \
    int r = rank64(ksq);                                          \
    const bool flip_file = (f < 4);                               \
    if(flip_rank) r = 7 - r;                                      \
    if(flip_file) f = 7 - f;                                      \
    const unsigned kidx = (r * 4 + (f - 4));

/*
Recompute accumulator from scratch
*/
INLINE void recalculate_accumulator(int side, input_weight_t* const accumulation,
  Position* pos)
{
    //king index
    KINDEX();

    //initialize accumulater
    memcpy(accumulation, input_biases, sizeof(input_biases));

    //add contributions of each piece
    for(unsigned i = 0, pc; (pc = pos->pieces[i]) != 0; i++) {
        unsigned sq = pos->squares[i];
        INFLUENCE(add);
    }
}
/*
Update accumulator from history of moves
*/
INLINE void update_accumulator(int side, input_weight_t* const accumulation,
  Position* pos, const DirtyPiece* const dp)
{
    //king index
    KINDEX();

    //dirty piece loop
    for (int i = 0; i < dp->dirtyNum; i++) {
      unsigned pc, sq;

      //delete from piece
      pc = dp->pc[i];
      sq = dp->from[i];
      if (sq != 64) {
        INFLUENCE(sub);
      }

      //add to piece
      pc = dp->pc[i];
      sq = dp->to[i];
      if (sq != 64) {
        INFLUENCE(add);
      }
    }
}
/*
Input layer computation
*/
INLINE void INPUT_LAYER(Position *pos, output_t* const output)
{
    Accumulator* const accumulator = &(pos->nncpu[0]->accumulator);

    //compute accumulator
    if (!accumulator->computedAccumulation) {
        Accumulator *pacc;
        if (   (!pos->nncpu[1] || !(pacc = &pos->nncpu[1]->accumulator)->computedAccumulation)
            && (!pos->nncpu[2] || !(pacc = &pos->nncpu[2]->accumulator)->computedAccumulation)
        ) {
            //if no previous accumulation, recalculate from scratch
            for(unsigned side = 0; side < 2; side++)
              recalculate_accumulator(side, accumulator->accumulation[side], pos);
        } else {
            //update accumulator
            const DirtyPiece *dp = &(pos->nncpu[0]->dirtyPiece);
            if(pos->nncpu[1]->accumulator.computedAccumulation) {
              for(unsigned side = 0; side < 2; side++) {
                  if(dp->pc[0] == (int)KING(side)) {
                    recalculate_accumulator(side, accumulator->accumulation[side], pos);
                  } else {
                    memcpy(accumulator->accumulation[side],
                      pacc->accumulation[side], sizeof(input_weight_t)*256);
                    update_accumulator(side, accumulator->accumulation[side], pos, dp);
                  }
              }
            } else {
              const DirtyPiece *dp2 = &(pos->nncpu[1]->dirtyPiece);
              for(unsigned side = 0; side < 2; side++) {
                  if(dp->pc[0] == (int)KING(side) || dp2->pc[0] == (int)KING(side)) {
                    recalculate_accumulator(side, accumulator->accumulation[side], pos);
                  } else {
                    memcpy(accumulator->accumulation[side],
                      pacc->accumulation[side], sizeof(input_weight_t)*256);
                    update_accumulator(side, accumulator->accumulation[side], pos, dp);
                    update_accumulator(side, accumulator->accumulation[side], pos, dp2);
                  }
              }
            }
        }
    }

    //assing scaled accumulation to output nodes
    clampVector<256,input_weight_t,output_t>(accumulator->accumulation[pos->player],output);
    clampVector<256,input_weight_t,output_t>(accumulator->accumulation[1 - pos->player],output + 256);

    accumulator->computedAccumulation = 1;
}
/*
evaluate net
*/
int nncpu_evaluate_pos(Position* pos)
{
    //output_t
    CACHE_ALIGN output_t input_output[2*256];
    CACHE_ALIGN output_t hidden1_output[32];
    CACHE_ALIGN output_t hidden2_output[32];
    CACHE_ALIGN bias_t temp[32];
    bias_t score[1];

    //accumulate player and opponent heads
    INPUT_LAYER(pos,input_output);

    //three dense layers
    DENSE<512,32>(input_output,hidden1_weights,hidden1_biases,temp);
    clampVector<32,bias_t,output_t>(temp,hidden1_output);
    DENSE<32,32>(hidden1_output,hidden2_weights,hidden2_biases,temp);
    clampVector<32,bias_t,output_t>(temp,hidden2_output);
    DENSE<32,1>(hidden2_output,output_weights,output_biases,score);

    return score[0] / (0.00575646273 * SCALE_BIAS);
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
Evaluate position
*/
DLLExport int _CDECL nncpu_evaluate(
  int player, int* pieces, int* squares)
{
  NNCPUdata nncpu;
  nncpu.accumulator.computedAccumulation = 0;

  Position pos;
  pos.nncpu[0] = &nncpu;
  pos.nncpu[1] = 0;
  pos.nncpu[2] = 0;
  pos.player = player;
  pos.pieces = pieces;
  pos.squares = squares;
  return nncpu_evaluate_pos(&pos);
}
/*
Incremental evaluation
*/
DLLExport int _CDECL nncpu_evaluate_incremental(
  int player, int* pieces, int* squares, NNCPUdata** nncpu)
{
  assert(nncpu[0] && uint64_t(&nncpu[0]->accumulator) % 64 == 0);

  Position pos;
  pos.nncpu[0] = nncpu[0];
  pos.nncpu[1] = nncpu[1];
  pos.nncpu[2] = nncpu[2];
  pos.player = player;
  pos.pieces = pieces;
  pos.squares = squares;
  return nncpu_evaluate_pos(&pos);
}
/*
Decode fen
*/
static const char piece_name[] = "_KQRBNPkqrbnp_";
static const char rank_name[] = "12345678";
static const char file_name[] = "abcdefgh";
static const char col_name[] = "WwBb";
static const char cas_name[] = "KQkq";

static void decode_fen(const char* fen_str, int* player, int* castle,
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
/*
Evaluate fen
*/
DLLExport int _CDECL nncpu_evaluate_fen(const char* fen)
{
  int pieces[33],squares[33],player,castle,fifty,move_number;
  decode_fen((char*)fen,&player,&castle,&fifty,&move_number,pieces,squares);;
  return nncpu_evaluate(player,pieces,squares);
}