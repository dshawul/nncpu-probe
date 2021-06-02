#include <assert.h>
#include <inttypes.h>
#include <string.h>
#include <stdio.h>

#define DLL_EXPORT
#include "nncpu.h"
#undef DLL_EXPORT
#include "misc.h"

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
types and quantization
*/
#if !defined(USE_FLOAT)

#define SCALE_WEIGHT_SHIFT  6
#define SCALE_WEIGHT        (1 << SCALE_WEIGHT_SHIFT)
#define SCALE_ACT           127
#define SCALE_BIAS          (SCALE_WEIGHT*SCALE_ACT)

typedef int8_t  weight_t;
typedef int32_t bias_t;
typedef int8_t  output_t;
typedef int16_t input_weight_t;

#else

#define SCALE_WEIGHT_SHIFT  0
#define SCALE_WEIGHT        1
#define SCALE_ACT           1
#define SCALE_BIAS          1

typedef float weight_t;
typedef float bias_t;
typedef float output_t;
typedef float input_weight_t;

#endif

/*
SIMD optimized operations
*/
#include "simd.h"

/*
weights and biases
*/
#define CACHE_ALIGN alignas(64)

#ifdef STOCK
CACHE_ALIGN static input_weight_t input_weights[N_K_INDICES*(10*64+1)*256]; //order: [N_inp][N_out]
#else
CACHE_ALIGN static input_weight_t input_weights[N_K_INDICES*(12*64+0)*256]; //order: [N_inp][N_out]
#endif
CACHE_ALIGN static input_weight_t input_biases[256];

CACHE_ALIGN static weight_t hidden1_weights[32*512];           //order: [N_out][N_inp]
CACHE_ALIGN static bias_t hidden1_biases[32];

CACHE_ALIGN static weight_t hidden2_weights[32*32];            //order: [N_out][N_inp]
CACHE_ALIGN static bias_t hidden2_biases[32];

CACHE_ALIGN static weight_t output_weights[1*32];              //order: [N_out][N_inp]
CACHE_ALIGN static bias_t output_biases[1];

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
#ifdef STOCK

//mapping tables
enum {
  PS_W_PAWN,   PS_B_PAWN,
  PS_W_KNIGHT, PS_B_KNIGHT,
  PS_W_BISHOP, PS_B_BISHOP,
  PS_W_ROOK,   PS_B_ROOK,
  PS_W_QUEEN,  PS_B_QUEEN,
};

uint32_t piece_map[2][14] = {
  { 0, 0, PS_W_QUEEN, PS_W_ROOK, PS_W_BISHOP, PS_W_KNIGHT, PS_W_PAWN,
       0, PS_B_QUEEN, PS_B_ROOK, PS_B_BISHOP, PS_B_KNIGHT, PS_B_PAWN, 0},
  { 0, 0, PS_B_QUEEN, PS_B_ROOK, PS_B_BISHOP, PS_B_KNIGHT, PS_B_PAWN,
       0, PS_W_QUEEN, PS_W_ROOK, PS_W_BISHOP, PS_W_KNIGHT, PS_W_PAWN, 0}
};

//add/subtract influence of input node
#define INFLUENCE(op,kidx_) {                                     \
  if(flip_rank) sq = sq ^ 0x3f;                                   \
  input_weight_t* const pinp = input_weights +                    \
        ((kidx_)*641 + (piece_map[side][pc]*64+1) + sq) * 256;    \
  op##Vectors<256,input_weight_t>(accumulation,pinp);             \
}

//compute king index (0 to 63)
#define KINDEX()                                                  \
    const bool flip_rank = (side == 1);                           \
    const unsigned ksq = pos->squares[side];                      \
    const unsigned kidx = (flip_rank ? (ksq ^ 0x3f) : ksq);

#else //STOCK

#define MIRRORF64(sq)    ((sq) ^ 007)
#define MIRRORR64(sq)    ((sq) ^ 070)
#define file64(x)        ((x) &  7)
#define rank64(x)        ((x) >> 3)
#define INVERT(x)        (((x) > 6) ? ((x) - 6) : ((x) + 6))

//add/subtract influence of input node
#define INFLUENCE(op,kidx_) {                                     \
  if(flip_rank) {                                                 \
      sq = MIRRORR64(sq);                                         \
      pc = INVERT(pc);                                            \
  }                                                               \
  if(flip_file) {                                                 \
      sq = MIRRORF64(sq);                                         \
  }                                                               \
  input_weight_t* const pinp = input_weights +                    \
                ((kidx_)*768 + (pc-1)*64 + sq) * 256;             \
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
    const unsigned kidx = KINDEX[(r * 4 + (f - 4))];

static const unsigned KINDEX[] = {
#if N_K_INDICES==32
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  9, 10, 11,
   12, 13, 14, 15,
   16, 17, 18, 19,
   20, 21, 22, 23,
   24, 25, 26, 27,
   28, 29, 30, 31
#elif N_K_INDICES==16
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  8,  9,  9,
   10, 10, 11, 11,
   12, 12, 13, 13,
   12, 12, 13, 13,
   14, 14, 15, 15,
   14, 14, 15, 15
#elif N_K_INDICES==8
    0,  1,  2,  3,
    4,  4,  5,  5,
    6,  6,  6,  6,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7,
    7,  7,  7,  7
#elif N_K_INDICES==4
    0,  0,  1,  1,
    2,  2,  2,  2,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3,
    3,  3,  3,  3
#elif N_K_INDICES==2
    0,  0,  0,  0,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1,
    1,  1,  1,  1
#elif N_K_INDICES==1
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0,
    0,  0,  0,  0
#endif
};
#endif //STOCK

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
#ifdef STOCK
    for(unsigned i = 2, pc; (pc = pos->pieces[i]) != 0; i++) {
#else
    for(unsigned i = 0, pc; (pc = pos->pieces[i]) != 0; i++) {
#endif
        unsigned sq = pos->squares[i];
        INFLUENCE(add,kidx);
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
#ifdef STOCK
      if(pc == wking || pc == bking)
        continue;
#endif
      sq = dp->from[i];
      if (sq != 64) {
        INFLUENCE(sub,kidx);
      }

      //add to piece
      pc = dp->pc[i];
      sq = dp->to[i];
      if (sq != 64) {
        INFLUENCE(add,kidx);
      }
    }
}
/*
Input layer computation
*/
#define KING(c) ((c) ? bking : wking)

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

#ifdef STOCK
    return (int)(score[0] / 16.0);
#else
    return (int)(score[0] / (0.00575646273 * SCALE_BIAS));
#endif
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
Evaluate fen
*/
DLLExport int _CDECL nncpu_evaluate_fen(const char* fen)
{
  int pieces[33],squares[33],player,castle,fifty,move_number;
  decode_fen((char*)fen,&player,&castle,&fifty,&move_number,pieces,squares);;
  return nncpu_evaluate(player,pieces,squares);
}
/*
Read bytes in little endian byte order
*/
static uint32_t read_bytes(int count,FILE* f) {
    uint32_t x = 0;
    uint8_t* c = (uint8_t*) &x;
    for(int i = 0; i < count; i++)
        c[i] = ((uint8_t) fgetc(f));
    return x;
}
static float read_bytes_f(int count,FILE* f) {
    uint32_t x = 0;
    uint8_t* c = (uint8_t*) &x;
    for(int i = 0; i < count; i++)
        c[i] = ((uint8_t) fgetc(f));
    float* p = (float*) &x;
    return *p;
}
#ifdef STOCK
static void skip_bytes(int count,FILE* f) {
    for(int i = 0; i < count; i++)
        fgetc(f);
}
#endif
/*
Read weights.
   shuffle order of weights for best cache performance
*/
#ifdef STOCK
static bool read_network(FILE* f)
{
    uint32_t val;

    //header
    val = read_bytes(sizeof(int),f);
    if(val != 0x7AF32F16u) return false;

    val = read_bytes(sizeof(int),f);
    if(val != 0x3e5aa6eeU) return false;

    val = read_bytes(sizeof(int),f);
    if(val != 177) return false;

    skip_bytes(177,f);

    //------- transformer ---------
    val = read_bytes(sizeof(int),f);
    if(val != 0x5d69d7b8) return false;

    //read input weights and biases
    for(int i = 0; i < 256; i++)
        input_biases[i] = read_bytes(sizeof(uint16_t), f);
    for(int i = 0; i < N_K_INDICES * 641 * 256; i++)
        input_weights[i] = read_bytes(sizeof(uint16_t), f);

    //-------- dense network 512x32x32x1 ----------
    val = read_bytes(sizeof(int),f);
    if(val != 0x63337156) return false;

    //read hidden1 weights and biases
    for(int i = 0; i < 32; i++)
        hidden1_biases[i] = read_bytes(sizeof(uint32_t), f);
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 512; j++) {
            hidden1_weights[i*512 + j] = read_bytes(sizeof(uint8_t), f);
        }
    }
    //read hidden2 weights and biases
    for(int i = 0; i < 32; i++)
        hidden2_biases[i] = read_bytes(sizeof(uint32_t), f);
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 32; j++) {
            hidden2_weights[i*32 + j] = read_bytes(sizeof(uint8_t), f);
        }
    }
    //read output weights and biases
    for(int i = 0; i < 1; i++)
        output_biases[i] = read_bytes(sizeof(uint32_t), f);
    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 1; j++) {
            output_weights[i*1 + j] = read_bytes(sizeof(uint8_t), f);
        }
    }

    return true;
}
#else
static bool read_network(FILE* f)
{

    //version number
    uint32_t ver;
    ver = read_bytes(sizeof(uint32_t),f);
    if(ver != 0x00000000) return false;

    //input layer
    for(int sq = 0; sq < 64; sq++) {
        for(int kidx = 0; kidx < N_K_INDICES; kidx++) {
            for(int pc = 0; pc < 12; pc++) {
                for(int o = 0; o < 256; o++) {
                    float value = read_bytes_f(sizeof(float), f) * SCALE_BIAS;
                    input_weights[kidx*12*64*256 + pc*64*256 + sq*256+o] =
                        (input_weight_t)value;
                }
            }
        }
    }
    for(int o = 0; o < 256; o++) {
        float value = read_bytes_f(sizeof(float), f) * SCALE_BIAS;
        input_biases[o] = (input_weight_t)value;
    }

    //first hidden layer
    for(int i = 0; i < 512; i++) {
        for(int j = 0; j < 32; j++) {
            float value = read_bytes_f(sizeof(float), f) * SCALE_WEIGHT;
            hidden1_weights[j*512 + i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 32; i++) {
        float value = read_bytes_f(sizeof(float), f) * SCALE_BIAS;
        hidden1_biases[i] = (bias_t)value;
    }

    //second hidden layer
    for(int i = 0; i < 32; i++){
        for(int j = 0; j < 32; j++){
            float value = read_bytes_f(sizeof(float), f) * SCALE_WEIGHT;
            hidden2_weights[j*32+i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 32; i++) {
        float value = read_bytes_f(sizeof(float), f) * SCALE_BIAS;
        hidden2_biases[i] = (bias_t)value;
    }

    //output layer
    for(int i = 0; i < 32; i++) {
        for(int j =0; j < 1; j++) {
            float value = read_bytes_f(sizeof(float), f) * SCALE_WEIGHT;
            output_weights[j*32+i] = (weight_t)value;
        }
    }
    for(int i = 0; i < 1; i++) {
        float value = read_bytes_f(sizeof(float), f) * SCALE_BIAS;
        output_biases[i] = (bias_t)value;
    }

    return true;
}
#endif
/*
init net
*/
DLLExport void _CDECL nncpu_init(const char* path)
{
    FILE* f = fopen(path, "rb");
    if(!f) {
        printf("*******  NNCPU file not found! *******\n");
        fflush(stdout);
        return;
    } else {
        printf("Loading NNCPU : %s\n", path);
        fflush(stdout);
    }
    if(!read_network(f)) {
       fclose(f);
       printf("********* Network failed to load properly! *******\n");
       fflush(stdout);
       return;
    }
    fclose(f);

    printf("NNCPU loaded !\n");
    fflush(stdout);
}
